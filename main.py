import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

FILE_PATH = "AR_2026-04-01.pdf"
CHROMA_PATH = "chroma_db" # Čia bus saugomi tavo vektoriai

def ingest_data():
    # 1. Užkrauname ir skaidome (jau žinai, kaip tai veikia)
    loader = PyPDFLoader(FILE_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Paruošta įrašymui: {len(chunks)} fragmentų.")

    # 2. Inicijuojame Embedding modelį (paverčia tekstą skaičiais)
    print("Kraunamas AI modelis vektorizavimui (tai gali užtrukti pirmą kartą)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Sukuriame ChromaDB ir išsaugome vektorius
    print("Vektorizuojama ir saugoma į ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"Sėkmingai sukurta vektorinė bazė aplanke: {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_data()