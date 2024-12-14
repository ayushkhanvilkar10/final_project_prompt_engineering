# create_database.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

# Load environment variables
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_FILE = "data/nyc/nyc.txt"

def main():
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {os.path.abspath(DATA_FILE)}")
        print("Current working directory:", os.getcwd())
        print("\nPlease ensure the file exists at the correct location.")
        return
    
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    if not documents:
        print("No documents were loaded. Cannot proceed.")
        return
        
    chunks = split_text(documents)
    if not chunks:
        print("No chunks were created. Cannot proceed.")
        return
        
    save_to_chroma(chunks)

def load_documents():
    try:
        print(f"Attempting to load: {os.path.abspath(DATA_FILE)}")
        loader = TextLoader(DATA_FILE)
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} documents")
        return documents
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        return []

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    if chunks:
        print("\nSample chunk:")
        print("-------------")
        print(f"Content: {chunks[0].page_content[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
    
    return chunks

def save_to_chroma(chunks: list[Document]):
    if not chunks:
        print("No chunks to save to Chroma.")
        return
        
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()