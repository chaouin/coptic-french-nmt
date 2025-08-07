import argparse
import os

import pandas as pd
from lxml import etree

# === Dictionary mapping bnumber to book name ===
BOOK_ID_MAP = {
    "1": "Genesis", "2": "Exodus", "3": "Leviticus", "4": "Numbers", "5": "Deuteronomy",
    "6": "Joshua", "7": "Judges", "8": "Ruth", "9": "I Samuel", "10": "II Samuel",
    "11": "I Kings", "12": "II Kings", "13": "I Chronicles", "14": "II Chronicles",
    "17": "Esther", "18": "Job", "19": "Psalms", "20": "Proverbs", "21": "Ecclesiastes",
    "22": "Song of Solomon", "23": "Isaiah", "24": "Jeremiah", "25": "Lamentations",
    "26": "Ezekiel", "27": "Daniel", "28": "Hosea", "29": "Joel", "30": "Amos",
    "31": "Obadiah", "32": "Jonah", "33": "Micah", "34": "Nahum", "35": "Habakkuk",
    "36": "Zephaniah", "37": "Haggai", "38": "Zechariah",
    "40": "Matthew", "41": "Mark", "42": "Luke", "43": "John", "44": "Acts of the Apostles",
    "45": "Romans", "46": "1 Corinthians", "47": "2 Corinthians", "48": "Galantians",
    "49": "Ephesians", "50": "Philippians", "51": "Colossians", "52": "1 Thessalonians",
    "53": "2 Thessalonians", "54": "1 Timothy", "55": "2 Timothy", "56": "Titus",
    "57": "Philemon", "58": "Hebrew", "59": "James", "60": "1 Peter", "61": "2 Peter",
    "62": "1 John", "63": "2 John", "64": "3 John", "65": "Jude", "66": "Revelation",
    "67": "Tobit", "71": "Sirach"
}

# === 1. Load all French verses (Zefania) ===
def extract_all_french_verses(zefania_file):
    tree = etree.parse(zefania_file)
    verses = {}
    for book in tree.xpath("//BIBLEBOOK"):
        bnum = book.attrib["bnumber"]
        if bnum in BOOK_ID_MAP:
            book_name = BOOK_ID_MAP[bnum]
            for chapter in book.xpath("./CHAPTER"):
                cnum = int(chapter.attrib["cnumber"])
                for verse in chapter.xpath("./VERS"):
                    vnum = int(verse.attrib["vnumber"])
                    verse_id = f"{book_name} {cnum}.{vnum}"
                    verses[verse_id] = verse.text
    return verses

# === MAIN ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the Coptic–French parallel corpus for a given version.")
    parser.add_argument("--zefania", required=True, help="Zefania XML file of the French version")
    parser.add_argument("--version", required=True, help="Name of the target version (e.g., darby, crampon, segond)")
    parser.add_argument("--coptic", default="coptic_corpus.csv", help="CSV file of the full Coptic corpus (NT + OT + Tobit)")
    parser.add_argument("--single_coptic", default="single_book_corpus.csv", help="CSV file of the single book corpus")
    parser.add_argument("--single_book_name", help="Name of the single book (e.g., 'Tobit', 'Sirach')")
    parser.add_argument("--only_single_book", action="store_true", help="Only process the specified single book")
    parser.add_argument("--translation_col", default="english_translation", help="Name of the translation column")

    args = parser.parse_args()

    # === Load Coptic corpus (based on mode) ===
    if args.only_single_book:
        if not args.single_book_name:
            parser.error("--single_book_name is required with --only_single_book.")
        print(f"=== SINGLE BOOK mode: {args.single_book_name} ===")
        df = pd.read_csv(args.single_coptic)
    else:
        print("=== FULL CORPUS mode (NT + OT + Tobit) ===")
        df = pd.read_csv(args.coptic)

    # === Extract French verses from Zefania ===
    french_verses = extract_all_french_verses(args.zefania)

    # === Add French translation column ===
    df["french_translation"] = df["verse_id"].map(french_verses)

    # === Determine save path ===
    output_folder = os.path.join(args.version)
    os.makedirs(output_folder, exist_ok=True)

    if args.only_single_book:
        output_file = os.path.join(output_folder, f"{args.single_book_name.lower()}_parallel_corpus_fr_{args.version}.csv")
    else:
        output_file = os.path.join(output_folder, f"coptic_parallel_corpus_fr_{args.version}.csv")

    # === Save enriched corpus ===
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ Enriched corpus with French verses saved to: {output_file}")
