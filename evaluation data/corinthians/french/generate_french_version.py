import csv

from lxml import etree


def extract_bible_book(xml_file, book_number, book_name, version_name, output_csv):
    """
    Extracts a specific book from a Zefania XML Bible file and writes it to a CSV.

    Parameters:
    xml_file (str): Path to the Bible XML file.
    book_number (str): ID of the book (e.g., "46" for 1 Corinthians).
    book_name (str): Name of the book to use in the verse_id (e.g., "1 Corinthians").
    version_name (str): Version identifier (e.g., "segond").
    output_csv (str): Name of the output CSV file.
    """
    tree = etree.parse(xml_file)
    root = tree.getroot()

    verses = []

    # Find <BIBLEBOOK bnumber="46">
    for book in root.xpath(f'//BIBLEBOOK[@bnumber="{book_number}"]'):
        for chapter in book.xpath('.//CHAPTER'):
            chapter_num = chapter.get("cnumber")
            for verse in chapter.xpath('./VERS'):
                verse_num = verse.get("vnumber")
                verse_text = verse.text.strip() if verse.text else ""
                verse_id = f"{book_name} {chapter_num}.{verse_num}"
                verses.append((verse_id, verse_text))

    # Write the CSV file
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["verse_id", f"french_{version_name}"])
        writer.writerows(verses)

    print(f"✅ {output_csv} generated with {len(verses)} verses.")

# === USAGE EXAMPLE ===
if __name__ == "__main__":
    extract_bible_book(
        xml_file="bible_segond.xml",
        book_number="46",
        book_name="1 Corinthians",
        version_name="segond",
        output_csv="1corinthians_segond.csv"
    )
    extract_bible_book(
        xml_file="bible_darby.xml",
        book_number="46",
        book_name="1 Corinthians",
        version_name="darby",
        output_csv="1corinthians_darby.csv"
    )
    extract_bible_book(
        xml_file="bible_crampon.xml",
        book_number="46",
        book_name="1 Corinthians",
        version_name="crampon",
        output_csv="1corinthians_crampon.csv"
    )
