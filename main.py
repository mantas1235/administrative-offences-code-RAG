import os
import shutil
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Naujausia biblioteka
from langchain_community.vectorstores import Chroma

FILE_PATH = "AR_2026-04-01.pdf"
CHROMA_PATH = "chroma_db"

def ingest_data():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Išvalyta sena duomenų bazė.")

    # 1. Kokybiškas PDF nuskaitymas
    loader = PDFPlumberLoader(FILE_PATH)
    documents = loader.load()
    
    # 2. Preciziškas skaidymas (dideli fragmentai geresniam kontekstui)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Paruošta {len(chunks)} fragmentų.")

    # 3. Galingas daugiakalbis modelis (išnaudoja tavo 32GB)
    model_name = "intfloat/multilingual-e5-large"
    print(f"Kraunamas sunkiasvoris modelis: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'} # Jei neturi NVIDIA GPU, naudojame CPU
    )

    # 4. Inicijuojame tuščią Chroma bazę
    vector_db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )

    # 5. Įrašymas dalimis (Batching) - Svarbiausia dalis!
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vector_db.add_documents(batch)
        print(f"Progresas: {i + len(batch)} / {len(chunks)} vektorizuota...")

    print("\n[FINISH] Vektorinė bazė sukurta sėkmingai!")

if __name__ == "__main__":
    ingest_data()