import os
import re

import pandas as pd
from lxml import etree

# === MAPPING FOR 1 CORINTHIANS ===
BOOK_ID_MAP_1COR = {"46": "1 Corinthians"}

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

# === CORE EXTRACTION: TRANSLATION ONLY ===
def build_translation_only_corpus(verse_path, translation_path, book_code, book_id_map):
    verse_ids = parse_feats(verse_path)
    translations = parse_feats(translation_path)

    book_name = book_id_map.get(book_code, f"Book{book_code}")
    data = []

    for span_id, verse_num in verse_ids.items():
        if span_id in translations:
            match = re.search(r'(\d+)$', verse_num)
            if not match:
                continue
            verse_number = match.group(1)

            # Assume chapter is included in verse string if formatted as chapter:verse
            match_full = re.search(r"(\d+):(\d+)", verse_num)
            if match_full:
                chapter = match_full.group(1)
                verse = match_full.group(2)
            else:
                # fallback: assume chapter 1
                chapter = "1"
                verse = verse_number

            verse_id = f"{book_name} {chapter}.{verse}"
            english_translation = translations[span_id]
            data.append((verse_id, english_translation))

    return sorted(data, key=lambda x: tuple(map(int, re.search(r"(\d+)\.(\d+)", x[0]).groups())))

# === PROCESS 1 CORINTHIANS ===
def process_1corinthians_translation_only(root_dir):
    all_data = []
    for book_code, book_name in BOOK_ID_MAP_1COR.items():
        all_chapter_dirs = [
            d for d in os.listdir(root_dir)
            if d.startswith("1Cor_") and os.path.isdir(os.path.join(root_dir, d))
        ]

        for folder_name in sorted(all_chapter_dirs):
            chapter_dir = os.path.join(root_dir, folder_name)
            try:
                corpus = build_translation_only_corpus(
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

# === MAIN ===
if __name__ == "__main__":
    print("=== Extracting English translations for 1 Corinthians ===")
    translation_data = process_1corinthians_translation_only("chapters")

    df = pd.DataFrame(translation_data, columns=["verse_id", "english_translation"])
    df.to_csv("1corinthians_english_translation.csv", index=False)
    print(f"\n✅ File generated: 1corinthians_english_translation.csv ({len(df)} verses)")
