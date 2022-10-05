from common import *

if __name__ == "__main__":
    con = sqlite3.connect("articles.db")
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS articles(name, url, date, filepath)")
    con.close()