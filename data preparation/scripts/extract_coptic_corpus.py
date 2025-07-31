import argparse
import re
import os
from lxml import etree

# === BOOK MAPPINGS ===
OLD_TESTAMENT_BOOK_ID_MAP = {
    "1": "Genesis", "2": "Exodus", "3": "Leviticus", "4": "Numbers", "5": "Deuteronomy",
    "6": "Joshua", "7": "Judges", "8": "Ruth", "9": "I Samuel", "10": "II Samuel",
    "11": "I Kings", "12": "II Kings", "13": "I Chronicles", "14": "II Chronicles",
    "17": "Esther", "18": "Job", "19": "Psalms", "20": "Proverbs", "21": "Ecclesiastes",
    "22": "Song of Solomon", "23": "Isaiah", "24": "Jeremiah", "25": "Lamentations",
    "26": "Ezekiel", "27": "Daniel", "28": "Hosea", "29": "Joel", "30": "Amos",
    "31": "Obadiah", "32": "Jonah", "33": "Micah", "34": "Nahum", "35": "Habakkuk",
    "36": "Zephaniah", "37": "Haggai", "38": "Zechariah"
}

BOOK_ID_MAP = {
    "40": "Matthew", "41": "Mark", "42": "Luke", "43": "John", "44": "Acts of the Apostles",
    "45": "Romans", "46": "1 Corinthians", "47": "2 Corinthians", "48": "Galantians",
    "49": "Ephesians", "50": "Philippians", "51": "Colossians", "52": "1 Thessalonians",
    "53": "2 Thessalonians", "54": "1 Timothy", "55": "2 Timothy", "56": "Titus",
    "57": "Philemon", "58": "Hebrew", "59": "James", "60": "1 Peter", "61": "2 Peter",
    "62": "1 John", "63": "2 John", "64": "3 John", "65": "Jude", "66": "Revelation"
}

TOBIT_BOOK_ID_MAP = { "67": "Tobit" }


# === PARSING FUNCTIONS ===
def parse_body_text(filepath):
    tree = etree.parse(filepath)
    return tree.findtext('.//body')

def parse_tokens(filepath):
    tree = etree.parse(filepath)
    token_map = {}
    pattern = re.compile(r"string-range\([^,]+,[^,]+,(\d+),(\d+)\)")
    for mark in tree.xpath("//mark"):
        token_id = mark.attrib["id"]
        href = mark.attrib["{http://www.w3.org/1999/xlink}href"]
        match = pattern.search(href)
        if match:
            start, length = map(int, match.groups())
            token_map[token_id] = (start, length)
    return token_map

def extract_token_text(token_map, body_text):
    return {
        token_id: body_text[start - 1:start - 1 + length]
        for token_id, (start, length) in token_map.items()
    }

def parse_span_tokens(filepath):
    tree = etree.parse(filepath)
    span_map = {}
    for mark in tree.xpath("//mark"):
        span_id = mark.attrib["id"]
        href = mark.attrib["{http://www.w3.org/1999/xlink}href"]
        tokens = [tok.strip("#") for tok in href.split()]
        span_map[span_id] = tokens
    return span_map

def parse_feats(filepath):
    tree = etree.parse(filepath)
    return {
        feat.attrib["{http://www.w3.org/1999/xlink}href"].strip("#"): feat.attrib["value"]
        for feat in tree.xpath("//feat")
    }

# === CORE FUNCTION ===
def build_parallel_corpus(body_path, token_path, mark_path, verse_path, translation_path, book_code, book_id_map):
    body_text = parse_body_text(body_path)
    token_map = parse_tokens(token_path)
    token_texts = extract_token_text(token_map, body_text)
    span_tokens = parse_span_tokens(mark_path)
    verse_ids = parse_feats(verse_path)
    translations = parse_feats(translation_path)

    book_name = book_id_map.get(book_code, f"Book{book_code}")

    chapter_match = re.search(r"_(\d+)\.text\.xml$", os.path.basename(body_path))
    chapter = int(chapter_match.group(1)) if chapter_match else 1

    data = []
    last_seen_verse = 0
    chapter_offset = 0

    for span_id, token_ids in span_tokens.items():
        if span_id in verse_ids and span_id in translations:
            raw_verse_number = verse_ids[span_id]

            if not re.fullmatch(r'\d+', raw_verse_number):
                print(f"⚠️ Verset ignoré : '{raw_verse_number}' dans {book_name} ({os.path.basename(body_path)})")
                continue

            verse_number = int(raw_verse_number)

            if verse_number <= last_seen_verse:
                chapter_offset += 1
            last_seen_verse = verse_number

            actual_chapter = chapter + chapter_offset
            verse_id = f"{book_name} {actual_chapter}.{verse_number}"
            coptic_text = ' '.join(token_texts[tid] for tid in token_ids if tid in token_texts)
            translation = translations[span_id]
            data.append((verse_id, coptic_text, translation))

    return sorted(data, key=lambda x: tuple(map(int, re.search(r"(\d+)\.(\d+)$", x[0]).groups())))

# === SAVE TO CSV ===
def save_to_csv(data, filename="coptic_corpus.csv", translation_col="english_translation"):
    import csv
    with open(filename, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["verse_id", "coptic_text", translation_col])
        writer.writerows(data)

# === PROCESS TESTAMENT ===
def process_testament(root_dir, book_id_map):
    all_data = []
    for book_code, book_name in book_id_map.items():
        book_prefix = f"{int(book_code):02}_{book_name.replace(' ', '_')}_"
        all_chapter_dirs = [
            d for d in os.listdir(root_dir)
            if d.startswith(book_prefix) and os.path.isdir(os.path.join(root_dir, d))
        ]

        for folder_name in sorted(all_chapter_dirs):
            chapter_dir = os.path.join(root_dir, folder_name)
            try:
                corpus = build_parallel_corpus(
                    body_path=os.path.join(chapter_dir, f"{folder_name}.text.xml"),
                    token_path=os.path.join(chapter_dir, f"{folder_name}.tok.xml"),
                    mark_path=os.path.join(chapter_dir, f"scriptorium.{folder_name}.mark.xml"),
                    verse_path=os.path.join(chapter_dir, f"scriptorium.{folder_name}.mark_verse_n.xml"),
                    translation_path=os.path.join(chapter_dir, f"scriptorium.{folder_name}.mark_translation.xml"),
                    book_code=book_code,
                    book_id_map=book_id_map
                )
                all_data.extend(corpus)
                print(f"✅ {folder_name} traité ({len(corpus)} versets)")
            except Exception as e:
                print(f"❌ Erreur dans {folder_name} : {e}")
    return all_data

# === MAIN ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction du corpus parallèle copte.")
    parser.add_argument("--single_book_dir",
                        help="Chemin vers le dossier du livre unique à traiter (ex: Sirach, Tobit, etc.)")
    parser.add_argument("--single_book_id", help="ID du livre unique (ex: 67 pour Tobit, 68 pour Sirach)")
    parser.add_argument("--single_book_name", help="Nom du livre unique (ex: 'Tobit', 'Sirach')")
    parser.add_argument("--single_output", default="single_book_corpus.csv",
                        help="Nom du fichier CSV de sortie pour le livre unique")
    parser.add_argument("--only_single_book", action="store_true", help="Ne traiter que le livre unique spécifié")
    parser.add_argument("--translation_col", default="english_translation", help="Nom de la colonne pour la traduction")

    args = parser.parse_args()

    if args.only_single_book:
        if not args.single_book_dir or not args.single_book_id or not args.single_book_name:
            parser.error(
                "--single_book_dir, --single_book_id et --single_book_name sont requis avec --only_single_book.")

        print(f"=== Traitement SEULEMENT Livre : {args.single_book_name} (ID {args.single_book_id}) ===")
        SINGLE_BOOK_ID_MAP = {args.single_book_id: args.single_book_name}
        all_data_single = process_testament(args.single_book_dir, SINGLE_BOOK_ID_MAP)
        save_to_csv(all_data_single, filename=args.single_output, translation_col=args.translation_col)
        print(f"✅ Corpus {args.single_book_name} généré : {args.single_output} ({len(all_data_single)} versets)")

    else:
        print("=== Traitement Nouveau Testament ===")
        all_data_nt = process_testament(args.newtestament, BOOK_ID_MAP)

        print("=== Traitement Ancien Testament ===")
        all_data_ot = process_testament(args.oldtestament, OLD_TESTAMENT_BOOK_ID_MAP)

        all_data = all_data_nt + all_data_ot
        save_to_csv(all_data, filename=args.output, translation_col=args.translation_col)
        print(f"✅ Corpus NT + OT généré : {args.output} ({len(all_data)} versets)")

        print("=== Traitement Livre de Tobit ===")
        TOBIT_BOOK_ID_MAP = {"67": "Tobit"}
        all_data_tobit = process_testament(args.tobit, TOBIT_BOOK_ID_MAP)
        save_to_csv(all_data_tobit, filename=args.tobit_output, translation_col=args.translation_col)
        print(f"✅ Corpus Tobit généré : {args.tobit_output} ({len(all_data_tobit)} versets)")

# ===== EXEMPLE LANCER LE SCRIPT ======
# python3 extract_coptic_corpus.py --newtestament "../raw data coptic/NewTestament" --oldtestament "../raw data coptic/OldTestament" --output "coptic_corpus.csv"
