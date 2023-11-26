# Chroma compatibility issues, hacking per its documentation
# https://docs.trychroma.com/troubleshooting#sqlite
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from typing import List

from tempfile import NamedTemporaryFile

import chainlit as cl
from chainlit.types import AskFileResponse
import chromadb
from chromadb.config import Settings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PDFPlumberLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from prompts import EXAMPLE_PROMPT, PROMPT


WELCOME_MESSAGE = """\
Welcome to Introduction to LLM App Development Sample PDF QA Application!
To get started:
1. Upload a PDF or text file
2. Ask any question about the file!
"""


def process_file(*, file: AskFileResponse) -> List[Document]:
    if file.type != "application/pdf":
        raise TypeError("Only PDF files are supported")

    with NamedTemporaryFile() as tempfile:
        tempfile.write(file.content)

        ######################################################################
        #
        # 1. Load the PDF
        #
        ######################################################################
        loader = PDFPlumberLoader(tempfile.name)
        ######################################################################
        documents = loader.load()

        ######################################################################
        #
        # 2. Split the text
        #
        ######################################################################
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        ######################################################################
        docs = text_splitter.split_documents(documents)

        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"

        if not docs:
            raise ValueError("PDF file parsing failed.")

        return docs


def create_search_engine(*, docs: List[Document], embeddings: Embeddings) -> VectorStore:
    # Initialize Chromadb client to enable resetting and disable telemtry
    client = chromadb.EphemeralClient()
    client_settings = Settings(
        chroma_db_impl="duckdb+parquet", anonymized_telemetry=False, persist_directory=".chromadb", allow_reset=True
    )

    # Reset the search engine to ensure we don't use old copies.
    # NOTE: we do not need this for production
    search_engine = Chroma(client=client, client_settings=client_settings)
    search_engine._client.reset()

    ##########################################################################
    #
    # 4. Create the document search engine. Remember to add
    # client_settings using the above settings.
    #
    ##########################################################################
    search_engine = Chroma.from_documents(
        client=client, documents=docs, embedding=embeddings, client_settings=client_settings
    )
    ##########################################################################

    return search_engine


@cl.on_chat_start
async def on_chat_start():
    # Asking user to to upload a PDF to chat with
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=WELCOME_MESSAGE,
            accept=["application/pdf"],
            max_size_mb=20,
        ).send()
    file = files[0]

    # Process and save data in the user session
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()
    docs = process_file(file=file)
    cl.user_session.set("docs", docs)
    msg.content = f"`{file.name}` processed. Loading ..."
    await msg.update()

    ##########################################################################
    #
    # 3. Set the Encoder model for creating embeddings
    #
    ##########################################################################
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    ##########################################################################
    try:
        search_engine = await cl.make_async(create_search_engine)(docs=docs, embeddings=embeddings)
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()
        raise SystemError
    msg.content = f"`{file.name}` loaded. You can now ask questions!"
    await msg.update()

    llm = ChatOpenAI(model="gpt-3.5-turbo-16k-0613", temperature=0, streaming=True)

    ##########################################################################
    #
    # 5. Create the chain / tool for RetrievalQAWithSourcesChain.
    #
    ##########################################################################
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=search_engine.as_retriever(max_tokens_limit=4097),
        ######################################################################
        # 6. Customize prompts to improve summarization and question
        # answering performance. Perhaps create your own prompt in prompts.py?
        ######################################################################
        chain_type_kwargs={"prompt": PROMPT, "document_prompt": EXAMPLE_PROMPT},
    )
    ##########################################################################

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    response = await chain.acall(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)]
    )

    answer = response["answer"]
    sources = response["sources"].strip()
    source_elements = []

    # Get the documents from the user session
    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    # Adding sources to the answer
    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=source_elements).send()
