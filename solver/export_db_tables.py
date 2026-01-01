import sqlite3
import csv
import os

DB_PATH = r"C:\Users\shute\cryptic-helper\scraper\cryptic.db"

# The tables you want to export
TABLES = [
    "indicators",
    "substitutions",
    "synonyms",
    "wordplay"
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "exported_tables")


def export_table(cursor, table_name):
    print(f"Exporting table: {table_name}")

    cursor.execute(f"PRAGMA table_info({table_name})")
    columns_info = cursor.fetchall()
    columns = [col[1] for col in columns_info]

    output_path = os.path.join(OUTPUT_DIR, f"{table_name}.csv")

    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    print(f" → Saved: {output_path}")


def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Loop over tables and export each
    for table in TABLES:
        try:
            export_table(cur, table)
        except Exception as e:
            print(f"Failed to export {table}: {e}")

    conn.close()
    print("\nDone exporting all tables.")


if __name__ == "__main__":
    main()
