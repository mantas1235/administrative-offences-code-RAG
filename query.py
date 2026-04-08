from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"

def search_in_db():
    # 1. Užkrauname tą patį modelį (kad skaičių kalba sutaptų)
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # 2. Prisijungiame prie egzistuojančios bazės
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 3. Vartotojo klausimas
    query = input("\nĮveskite klausimą apie ANK (pvz., baudos už greitį): ")

    # 4. Atliekame paiešką (ieškome 3 artimiausių fragmentų)
    results = db.similarity_search_with_relevance_scores(query, k=10)

    if len(results) == 0:
        print("Nepavyko rasti nieko panašaus.")
        return

    print(f"\n--- Rasti {len(results)} atitikmenys ---\n")
    for doc, score in results:
        print(f"Atitikimo balas: {score:.4f}")
        print(f"Fragmentas: {doc.page_content}")
        print("-" * 30)

if __name__ == "__main__":
    search_in_db()