import json
import time
import logging
import sqlite3
from datetime import datetime

from mw_api import fetch_synonyms_from_mw  # whatever your import is


DB_PATH = "cryptic.db"   # keep as-is if already defined
LOG_PATH = "fetch_synonyms.log"

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logging.info("=== fetch_synonyms.py STARTED ===")


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # --- how many rows are unfinished? ---
    cur.execute(
        """
        SELECT COUNT(*)
        FROM synonyms
        WHERE synonyms IS NULL
        """
    )
    remaining = cur.fetchone()[0]
    logging.info("Remaining words with synonyms IS NULL: %d", remaining)

    if remaining == 0:
        logging.info("Nothing to do. Exiting.")
        return

    # --- select batch ---
    cur.execute(
        """
        SELECT word
        FROM synonyms
        WHERE synonyms IS NULL
        ORDER BY word
        LIMIT 50
        """
    )
    rows = cur.fetchall()
    logging.info("Fetched %d rows for processing", len(rows))

    if not rows:
        logging.warning("Query returned zero rows despite remaining > 0")
        return

    for (word,) in rows:
        logging.info("Processing word: %s", word)

        try:
            syns, source = fetch_synonyms_from_mw(word)

            if syns is None:
                logging.warning("No response from API for word: %s", word)
                continue

            syn_json = json.dumps(syns)

            cur.execute(
                """
                UPDATE synonyms
                SET synonyms = ?, fetched_at = ?, source = ?
                WHERE word = ?
                """,
                (syn_json, datetime.utcnow(), source, word),
            )
            conn.commit()

            logging.info(
                "Stored %d synonyms for word: %s",
                len(syns),
                word,
            )

            time.sleep(1)  # keep your existing rate limit

        except Exception as e:
            logging.exception("ERROR processing word: %s", word)
            continue

    logging.info("=== fetch_synonyms.py FINISHED ===")


if __name__ == "__main__":
    main()
