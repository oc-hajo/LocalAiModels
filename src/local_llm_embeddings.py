import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import GPT4All

# Step 1: Load PDF Files
def load_pdfs_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    return documents

# Step 2: Compute Embeddings and Create Vectorstore
def create_vectorstore(documents, embedding_model_name="all-MiniLM-L6-v2", persist_directory=None):
    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    if persist_directory:
        vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=persist_directory)
    else:
        vectorstore = Chroma.from_documents(documents, embedding_model)
    return vectorstore

# Step 3: Initialize GPT4All and Create QA Chain
def initialize_qa_chain(vectorstore, model_path):
    llm = GPT4All(model=model_path)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Step 4: Query the Chain
def query_qa_chain(qa_chain, query):
    return qa_chain.run(query)

if __name__ == "__main__":
    # Directory containing PDF files
    pdf_directory = "files/"
    
    # Path to GPT4All model
    gpt4all_model_path = "/root/.cache/gpt4all/Llama-3.2-3B-Instruct-Q4_0.gguf"
    
    # Directory to persist embeddings (optional)
    persist_directory = "db"
    
    # Load documents from PDFs
    print("Loading PDF files...")
    documents = load_pdfs_from_directory(pdf_directory)
    
    # Compute embeddings and create vectorstore
    print("Creating vectorstore...")
    vectorstore = create_vectorstore(documents, persist_directory=persist_directory)
    
    # Initialize the QA chain
    print("Initializing QA chain...")
    qa_chain = initialize_qa_chain(vectorstore, gpt4all_model_path)
    
    # Query the chain
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Exiting...")
            break
        response = query_qa_chain(qa_chain, user_query)
        print("\nResponse:", response, "\n")
