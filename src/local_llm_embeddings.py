import asyncio
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import PromptTemplate
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.llm import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackManager


RETRIEVER = None
LLM = None

def log(text):
    print(text)

def load_documents():
    log("load documents")
    folder_path = "files"
    all_docs=[]
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            loader = PyPDFLoader(file_path=file_path)
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs


def init():
    global RETRIEVER, LLM
    file = "./chroma_db"
    embeddings = GPT4AllEmbeddings(model_name="/root/.cache/gpt4all/all-MiniLM-L6-v2.gguf2.f16.gguf")
    log("loading documents")
    if os.path.exists(file):
        vectorstore = Chroma(persist_directory=file, embedding_function=embeddings)
    else:
        docs = load_documents()
        log("split documents")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        log("save vectorestore to file")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=file)


    RETRIEVER = vectorstore.as_retriever()
    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    #retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")
    #print(retrieved_docs[0])
    #print("\n\n")
    #print(retrieved_docs[1])

    log("load llm")

    callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler()])
    LLM = GPT4All(model="/root/.cache/gpt4all/Llama-3.2-3B-Instruct-Q4_0.gguf", streaming=True, verbose=True,backend="llama", callback_manager=callback_manager)



TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
CUSTOM_RAG_PROMPT = PromptTemplate.from_template(TEMPLATE)

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


async def document_search(query:str):

    os.environ["OPENAI_API_KEY"] = "sk-s.."
    from langchain_openai import ChatOpenAI
    LLM = ChatOpenAI(model="gpt-3.5-turbo-0125")

    retrieved_docs = RETRIEVER.invoke(query)
    print(len(retrieved_docs))
    print(retrieved_docs[1])

    rag_chain = (
        {"context": RETRIEVER | format_docs, "question": RunnablePassthrough()}
        | CUSTOM_RAG_PROMPT
        | LLM
        | StrOutputParser()
    )

    print("lets go", flush=True)
    async for chunk in rag_chain.astream(query):
        print(chunk, end="|", flush=True)
        yield chunk

async def main():
    await document_search("What is Task Decomposition?")
    
if __name__ == "__main__":
    init()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
