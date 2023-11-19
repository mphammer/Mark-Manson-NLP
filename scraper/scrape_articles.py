from bs4 import BeautifulSoup
import requests, datetime 
import sqlite3
import os.path

MANSONNET = "https://markmanson.net"
DATABASE_NAME = "articles.db"

def get_all_article_metadata():
    article_archive_url = "{}{}".format(MANSONNET, "/archive")
    
    hdr = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
    }
    page = requests.get(article_archive_url, headers=hdr)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    main_content = soup.find('table', class_='archive-table')

    articles_metadata = {}
    tr_elements = main_content.find_all('tr')
    year = ""
    for tr_element in tr_elements:
        year_element = tr_element.find('div', class_='archive-year')
        if year_element:
            year = year_element.text
            continue 
        date_element = tr_element.find('span', class_='hide-on-mobile')
        date = date_element.text
        month_name, day = date.split(" ")
        datetime_object = datetime.datetime.strptime(month_name, "%B")
        month_number = datetime_object.month
        article_element = tr_element.find('a')
        path = "{}".format(article_element["href"])
        articles_metadata[path] = {
            "URL": "{}{}".format(MANSONNET, path),
            "Year": year,
            "Month": month_number, 
            "Day": day
        }
    
    return articles_metadata

def scrape_article(url, filename, data_path="../data/articles"):
    file_path = "{}/{}".format(data_path, filename)

    # Check Cache
    if os.path.isfile(file_path):
        return file_path

    # Scrape the webpage 
    hdr = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
    }
    page = requests.get(url, headers=hdr)
    soup = BeautifulSoup(page.content, 'html.parser')

    post_content = soup.find('div', class_='post-content')
    p_elements = post_content.find_all('p')

    with open(file_path, 'w') as f:
        for p_element in p_elements:
            if len(p_element.text) == 0 or p_element.text == "" or p_element.text == "\n":
                continue 
            f.write(p_element.text)
            f.write("\n")
    
    return file_path
    

def get_filename(year, month, day, text):
    return "{}-{}-{}-{}.txt".format(year, month, day, text)

if __name__ == "__main__":
    con = sqlite3.connect("../data/articles.db")
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS articles(name, url, date, filepath)")
    
    articles_metadata = get_all_article_metadata()
    for article_path, article_data in articles_metadata.items():
        article_path_clean = article_path.replace("/", "")
        filename = get_filename(article_data["Year"], article_data["Month"], article_data["Day"], article_path_clean)
        filepath = scrape_article(article_data["URL"], filename)
        date = "{}-{}-{}".format(article_data["Year"], article_data["Month"], article_data["Day"])
        query = "INSERT INTO articles (name, url, date, filepath) VALUES ('{}', '{}', '{}', '{}')".format(article_path_clean, article_data["URL"], date, filepath)
        cur.execute(query)
        con.commit()

    con.close()