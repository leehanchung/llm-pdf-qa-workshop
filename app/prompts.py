# flake8: noqa
from langchain.prompts import PromptTemplate

template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" field in your answer, with the format "SOURCES: <source1>, <source2>, <source3>, ...".

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"]
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)

WELCOME_MESSAGE = """\
Welcome to Introduction to LLM App Development Sample PDF QA Application!
To get started:
1. Upload a PDF or text file
2. Ask any question about the file!
"""
