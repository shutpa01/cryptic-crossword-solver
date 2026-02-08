import sqlite3
conn = sqlite3.connect(r'C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM indicators WHERE LOWER(word) = ?", ('leaders in',))
print(cursor.fetchall())
conn.close()