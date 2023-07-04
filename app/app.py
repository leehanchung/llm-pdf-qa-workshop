from tempfile import NamedTemporaryFile

from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI

import chainlit as cl
from chainlit.types import AskFileResponse
from chromadb.config import Settings


WELCOME_MESSAGE = """\
Welcome to Introduction to LLM App Development Sample PDF QA Application!
To get started:
1. Upload a PDF or text file
2. Ask any question about the file!
"""


def process_file(*, file: AskFileResponse) -> list:
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100
        )
        ######################################################################

        docs = text_splitter.split_documents(documents)

        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs


def create_search_engine(*, file: AskFileResponse) -> VectorStore:
    docs = process_file(file=file)

    
    ##########################################################################
    #
    # 3. Set the Encoder model for creating embeddings
    #
    ##########################################################################
    encoder = OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )
    ##########################################################################

    # Save data in the user session
    cl.user_session.set("docs", docs)

    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        anonymized_telemetry=False,
        persist_directory=".chromadb",
        allow_reset=True
    )

    ##########################################################################
    #
    # 4. Create the document search engine. Remember to add 
    # client_settings using the above settings.
    #
    ##########################################################################
    search_engine = Chroma.from_documents(
        documents=docs,
        embedding=encoder,
        metadatas=[doc.metadata for doc in docs],
        client_settings=client_settings 
    )
    ##########################################################################

    return search_engine


@cl.langchain_factory(use_async=True)
async def chat() -> Chain:

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=WELCOME_MESSAGE,
            accept=["application/pdf"],
            max_size_mb=20,
        ).send()
  
    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    try:
        search_engine = await cl.make_async(create_search_engine)(file=file)
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()

    llm = ChatOpenAI(
        model='gpt-3.5-turbo-16k-0613',
        temperature=0,
        streaming=True
    )

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
        # TODO: 6. Customize prompts to improve summarization and question
        # answering performance. Perhaps create your own prompt in prompts.py?
        ######################################################################
    )
    ##########################################################################

    await msg.update(content=f"`{file.name}` processed. You can now ask questions!")

    return chain


@cl.langchain_postprocess
async def process_response(res):
    answer = res["answer"]
    sources = res["sources"].strip()
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
