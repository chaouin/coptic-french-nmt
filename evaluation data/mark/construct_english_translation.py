import os
import re

import pandas as pd
from lxml import etree

# === BOOK MAPPING ===
BOOK_ID_MAP = {"41": "Mark"}

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
                continue

            verse_number = int(match.group(1))
            if verse_number <= last_seen_verse:
                chapter_offset += 1
            last_seen_verse = verse_number

            actual_chapter = chapter + chapter_offset
            verse_id = f"{book_name} {actual_chapter}.{verse_number}"
            coptic_text = ' '.join(token_texts[tid] for tid in token_ids if tid in token_texts)
            # Match with translation tokens
            verse_token_set = set(token_ids)
            matched_translation = ""
            for trans_id, trans_token_ids in span_tokens.items():
                if trans_id in translations:
                    if verse_token_set & set(trans_token_ids):  # non-empty intersection
                        matched_translation = translations[trans_id]
                        break

            data.append((verse_id, coptic_text, matched_translation))

    return sorted(data, key=lambda x: tuple(map(int, re.search(r"(\d+)\.(\d+)$", x[0]).groups())))

# === PROCESS BOOK ===
def process_book(root_dir):
    all_data = []
    for book_code, book_name in BOOK_ID_MAP.items():
        all_chapter_dirs = [
            d for d in os.listdir(root_dir)
            if d.startswith("41_Mark_") and os.path.isdir(os.path.join(root_dir, d))
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
                    book_id_map=BOOK_ID_MAP
                )
                all_data.extend(corpus)
                print(f"✅ {folder_name} processed ({len(corpus)} verses)")
            except Exception as e:
                print(f"❌ Error in {folder_name}: {e}")
    return all_data

# === MAIN ===
if __name__ == "__main__":
    print("=== Extracting Coptic–English parallel corpus for Mark ===")
    data = process_book("chapters")
    df = pd.DataFrame(data, columns=["verse_id", "coptic_text", "english_translation"])
    df.to_csv("mark_coptic_en.csv", index=False)
    print(f"✅ File generated: mark_coptic_en.csv ({len(df)} rows)")
