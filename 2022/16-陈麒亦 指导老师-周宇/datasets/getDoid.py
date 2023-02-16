import requests
import pandas
from scrapy import Selector
diseases=pandas.read_excel("./circR2Disease2.0/The circRNA-disease entries.xlsx")
x=set(diseases["Disease Name"].values)


def get_doid(i):
    headers = {
        "Referer": "https://disease-ontology.org/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46"

    }
    params = {
        "q": i,
        "adv_search": "False",
        "field-1": "name",
        "subset-1": "DO_AGR_slim",
        "relation-1": "adjacent+to",
        "tree": "obo"
    }
    html = requests.get("https://disease-ontology.org/search", headers=headers, params=params).text
    sel = Selector(text=html)
    doid = sel.xpath(".//tr[@class='search-row']/td/text()").extract()[:2]
    print(doid)
    return doid[0].split(":")[1]

for i in x:
    print(i)