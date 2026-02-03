import sqlite3
conn = sqlite3.connect(r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%indicator%'")
for row in cursor.fetchall():
    print(row[0])
conn.close()