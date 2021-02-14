import requests
from bs4 import BeautifulSoup


if __name__ == "__main__":
    target = 'https://www.xsbiquge.com/15_15338/8549128.html'
    req = requests.get(url=target)
    req.encoding = 'utf-8'
    html = req.text
    bs = BeautifulSoup(html, 'lxml')
    texts = bs.find('div', id='content')
    print(texts)
