from common import *

def get_best_article_names():
    best_articles_url = "{}{}".format(MANSONNET, "/best-articles")
    hdr = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
    }
    page = requests.get(best_articles_url, headers=hdr)
    soup = BeautifulSoup(page.content, 'html.parser')

    main_content = soup.find('div', class_='entry-content')
    li_elements = main_content.find_all('li')
    article_names = []
    for li_element in li_elements:
        a_element = li_element.find('a')
        link = a_element["href"]
        if link[0] == "#":
            continue 
        article_name = link.replace(MANSONNET,"")[1:] # remove leading "/"
        article_names.append(article_name)
    return article_names

if __name__ == "__main__":
    get_best_article_names()