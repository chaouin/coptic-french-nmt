import os
import re
import csv
import pandas as pd
from lxml import etree

# === BOOK MAPPINGS ===
BOOK_ID_MAP_1COR = {"48": "Galatians"}

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

# === CORE EXTRACTION ===
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
        if span_id in verse_ids:
            raw_verse_number = verse_ids[span_id]

            match = re.search(r'(\d+)$', raw_verse_number)
            if not match:
                print(f"⚠️ Skipped verse: '{raw_verse_number}' in {book_name} ({os.path.basename(body_path)})")
                continue

            verse_number = int(match.group(1))

            if verse_number <= last_seen_verse:
                chapter_offset += 1
            last_seen_verse = verse_number

            actual_chapter = chapter + chapter_offset
            verse_id = f"{book_name} {actual_chapter}.{verse_number}"
            coptic_text = ' '.join(token_texts[tid] for tid in token_ids if tid in token_texts)
            data.append((verse_id, coptic_text))

    return sorted(data, key=lambda x: tuple(map(int, re.search(r"(\d+)\.(\d+)$", x[0]).groups())))

# === PROCESS BOOK ===
def process_book(root_dir):
    all_data = []
    for book_code, book_name in BOOK_ID_MAP_1COR.items():
        all_chapter_dirs = [
            d for d in os.listdir(root_dir)
            if d.startswith("48_Galatians_") and os.path.isdir(os.path.join(root_dir, d))
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
                    book_id_map=BOOK_ID_MAP_1COR
                )
                all_data.extend(corpus)
                print(f"✅ {folder_name} processed ({len(corpus)} verses)")
            except Exception as e:
                print(f"❌ Error in {folder_name}: {e}")
    return all_data

# === MERGE WITH 3 TRANSLATIONS ===
def merge_with_french_versions(coptic_data, segond_csv, darby_csv, crampon_csv):
    df_coptic = pd.DataFrame(coptic_data, columns=["verse_id", "coptic_text"])
    df_segond = pd.read_csv(segond_csv)[["verse_id", "french_segond"]]
    df_darby  = pd.read_csv(darby_csv)[["verse_id", "french_darby"]]
    df_crampon= pd.read_csv(crampon_csv)[["verse_id", "french_crampon"]]

    # Progressive merge
    merged = df_coptic.merge(df_segond, on="verse_id", how="left")
    merged = merged.merge(df_darby, on="verse_id", how="left")
    merged = merged.merge(df_crampon, on="verse_id", how="left")

    merged.to_csv("galatians_coptic_3versions.csv", index=False)
    print(f"✅ Final file generated: galatians_coptic_3versions.csv ({len(merged)} rows)")

# === MAIN ===
if __name__ == "__main__":
    print("=== Extracting Coptic corpus for Galatians ===")
    data_1cor = process_book("chapters")

    # Provide your CSVs aligned by verse_id
    SEGOND_CSV = "french/galatians_segond.csv"
    DARBY_CSV  = "french/galatians_darby.csv"
    CRAMPON_CSV= "french/galatians_crampon.csv"

    merge_with_french_versions(data_1cor, SEGOND_CSV, DARBY_CSV, CRAMPON_CSV)
