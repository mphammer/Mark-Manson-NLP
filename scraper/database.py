import sqlite3
import numpy as np
import pandas as pd

if __name__ == "__main__":
    con = sqlite3.connect("../data/articles.db")
    cur = con.cursor()
    # cur.execute("CREATE TABLE IF NOT EXISTS articles(name, url, date, filepath)")
    # cur.execute("UPDATE articles SET filepath = REPLACE(filepath, 'articles/', '../data/articles/')")

    df = pd.read_sql_query("SELECT * from articles", con, index_col ="name")
    print(df.head())

    con.commit()
    con.close()