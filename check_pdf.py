from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./AR_2026-04-01.pdf")
pages = loader.load()

# Ieškome konkretaus raktinio žodžio visame 565 puslapių dokumente
target_word = "Viešosios rimties trikdymas"
found = False

for i, page in enumerate(pages):
    if target_word.lower() in page.page_content.lower():
        print(f"RADAU! Žodis '{target_word}' yra {i+1} puslapyje.")
        print("Turinio pavyzdys:")
        print(page.page_content[:300])
        found = True
        break

if not found:
    print("KLAIDA: PDF faile šio teksto nerasta. Gali būti, kad PDF yra skenuotas (nuotrauka).")