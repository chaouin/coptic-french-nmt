import csv

from lxml import etree


def extract_bible_book(xml_file, book_number, book_name, version_name, output_csv):
    """
    Extracts a specific book from a Zefania XML Bible file and saves it to a CSV.

    Parameters:
    - xml_file (str): Path to the Bible XML file
    - book_number (str): Book ID (e.g., "48" for Galatians)
    - book_name (str): Name of the book for the verse_id (e.g., "Galatians")
    - version_name (str): Version identifier (e.g., "segond")
    - output_csv (str): Output CSV filename
    """
    tree = etree.parse(xml_file)
    root = tree.getroot()

    verses = []

    # Find the <BIBLEBOOK bnumber="...">
    for book in root.xpath(f'//BIBLEBOOK[@bnumber="{book_number}"]'):
        for chapter in book.xpath('.//CHAPTER'):
            chapter_num = chapter.get("cnumber")
            for verse in chapter.xpath('./VERS'):
                verse_num = verse.get("vnumber")
                verse_text = verse.text.strip() if verse.text else ""
                verse_id = f"{book_name} {chapter_num}.{verse_num}"
                verses.append((verse_id, verse_text))

    # Write the CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["verse_id", f"french_{version_name}"])
        writer.writerows(verses)

    print(f"✅ {output_csv} generated with {len(verses)} verses.")

# === USAGE EXAMPLES ===
if __name__ == "__main__":
    ### GALATIANS ###
    extract_bible_book(
        xml_file="bible_segond.xml",
        book_number="48",
        book_name="Galatians",
        version_name="segond",
        output_csv="galatians_segond.csv"
    )
    extract_bible_book(
        xml_file="bible_darby.xml",
        book_number="48",
        book_name="Galatians",
        version_name="darby",
        output_csv="galatians_darby.csv"
    )
    extract_bible_book(
        xml_file="bible_crampon.xml",
        book_number="48",
        book_name="Galatians",
        version_name="crampon",
        output_csv="galatians_crampon.csv"
    )

    ### HEBREWS ###
    extract_bible_book(
        xml_file="bible_segond.xml",
        book_number="58",
        book_name="Hebrews",
        version_name="segond",
        output_csv="hebrews_segond.csv"
    )
    extract_bible_book(
        xml_file="bible_darby.xml",
        book_number="58",
        book_name="Hebrews",
        version_name="darby",
        output_csv="hebrews_darby.csv"
    )
    extract_bible_book(
        xml_file="bible_crampon.xml",
        book_number="58",
        book_name="Hebrews",
        version_name="crampon",
        output_csv="hebrews_crampon.csv"
    )

    ### MARK ###
    extract_bible_book(
        xml_file="bible_segond.xml",
        book_number="41",
        book_name="Mark",
        version_name="segond",
        output_csv="mark_segond.csv"
    )
    extract_bible_book(
        xml_file="bible_darby.xml",
        book_number="41",
        book_name="Mark",
        version_name="darby",
        output_csv="mark_darby.csv"
    )
    extract_bible_book(
        xml_file="bible_crampon.xml",
        book_number="41",
        book_name="Mark",
        version_name="crampon",
        output_csv="mark_crampon.csv"
    )
