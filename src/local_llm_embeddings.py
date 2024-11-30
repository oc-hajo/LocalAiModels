import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import PromptTemplate
import os
from langchain_community.document_loaders import PyPDFLoader


def log(text):
    print("\n\n\n"+text)

def load_documents():
    log("load documents")
    folder_path = "files"
    all_docs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            loader = PyPDFLoader(file_path=file_path)
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs


file = "./chroma_db"
if os.path.exists(file):
    vectorstore = Chroma(persist_directory=file, embedding_function=GPT4AllEmbeddings())
else:
    docs = load_documents()
    log("split documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    log("save vectorestore to file")
    vectorstore = Chroma.from_documents(documents=splits, embedding=GPT4AllEmbeddings(), persist_directory=file)





retriever = vectorstore.as_retriever()
#retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
#retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")
#print(retrieved_docs[0])
#print("\n\n")
#print(retrieved_docs[1])

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

log("load llm")

llm = GPT4All(model="/root/.cache/gpt4all/Llama-3.2-3B-Instruct-Q4_0.gguf")

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)


def document_search(query: str):
    print("document search going on")
    for chunk in rag_chain.stream(query):
        yield chunk