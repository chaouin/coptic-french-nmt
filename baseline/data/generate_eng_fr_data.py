import pandas as pd
from lxml import etree

# === Parser le XML
tree = etree.parse("bible_catholic_public_domain.xml")
root = tree.getroot()

# === Extraire chapitres
english_verses = {}
book_name = {"48": "Galatians", "58": "Hebrews", "46": "1 Corinthians", "41": "Mark" }

for book in root.findall(".//BIBLEBOOK"):
    book_id = book.get("bnumber")
    if book_id not in ["41", "46", "48", "58"]:
        continue

    book_label = book_name[book_id]
    for chapter in book.findall(".//CHAPTER"):
        chapter_num = chapter.get("cnumber")
        for verse in chapter.findall(".//VERS"):
            verse_num = verse.get("vnumber")
            verse_text = "".join(verse.itertext()).strip()
            verse_id = f"{book_label} {chapter_num}.{verse_num}"
            english_verses[verse_id] = verse_text

print(f"Total verses found: {len(english_verses)}")

# === Charger tes fichiers CSV
df = pd.read_csv("../../evaluation data/evaluation_data.csv")

# === Associer english_text
df["english_text"] = df["verse_id"].map(english_verses)

# === Garder colonnes demandées
final_df = df[["verse_id", "english_text", "french_segond", "french_darby", "french_crampon"]]

# === Sauvegarder
final_df.to_csv("evaluation_data_en_fr.csv", index=False)
print("✅ Fichier créé : evaluation_data_en_fr.csv")
