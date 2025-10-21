import os
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager



url = "https://repositorio-dspace.agricultura.gov.br/handle/1/137/simple-search?query=&filter_field_1=type&filter_type_1=equals&filter_value_1=Imagem&sort_by=score&order=desc&rpp=10&etal=0&start=0"
base_url = "https://repositorio-dspace.agricultura.gov.br"

def extract_images(td, base_url, driver, pasta_destino="intel/data/imagens"):
    link_tag = td.find("a")
    if not link_tag or not link_tag.has_attr("href"):
        return

    href = link_tag["href"]
    image_link = base_url + href
    print(f"\n→ Acessando item: {image_link}")

    driver.get(image_link)
    time.sleep(2)

    soup_item = BeautifulSoup(driver.page_source, "html.parser")

    tbodys = soup_item.find_all("tbody")
    if len(tbodys) < 2:
        print(" Segundo <tbody> não encontrado.")
        return

    tbody = tbodys[1]
    tr_com_link = None

    for tr in tbody.find_all("tr"):
        a_tag = tr.find("a")
        if a_tag and a_tag.has_attr("href"):
            tr_com_link = a_tag["href"]
            break

    if not tr_com_link:
        print(" Nenhum link encontrado no segundo <tbody>.")
        return

    download_page = base_url + tr_com_link
    print(f"  ↳ Indo para página de download: {download_page}")

    driver.get(download_page)
    time.sleep(2)

    soup_download = BeautifulSoup(driver.page_source, "html.parser")
    img_tag = soup_download.find("img")

    if not img_tag or not img_tag.has_attr("src"):
        print(" Nenhuma imagem encontrada nesta página.")
        return

    img_src = img_tag["src"]
    if not img_src.startswith("http"):
        img_src = base_url + img_src

    print(f"  ↓ Baixando imagem de: {img_src}")

    response = requests.get(img_src)
    if response.status_code == 200:
        filename = os.path.basename(img_src.split("?")[0])
        caminho = os.path.join(pasta_destino, filename)
        with open(caminho, "wb") as f:
            f.write(response.content)
        print(f"   Imagem salva em: {caminho}")
    else:
        print("Falha ao baixar imagem.")

def detect_next(soup):
    """
    Encontra o link da próxima página, se existir.
    O DSpace usa uma lista <ul class="pagination"> no final da página.
    """
    ul = soup.find("ul", class_="pagination")
    if not ul:
        return None

    li_list = ul.find_all("li")
    if not li_list:
        return None

    last_li = li_list[-1]
    a_tag = last_li.find("a")
    if a_tag and "href" in a_tag.attrs:
        if "disabled" not in last_li.get("class", []):
            return a_tag["href"]
    return None


driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

while True:
    driver.get(url)
    time.sleep(3)  

    soup = BeautifulSoup(driver.page_source, "html.parser")

    tbody = soup.select_one("table tbody")
    if not tbody:
        print(" Nenhum <tbody> encontrado nesta página.")
        break

    for tr in tbody.find_all("tr"):
        td = tr.find("td")
        if td:
            extract_images(td, base_url, driver)

    next_url = detect_next(soup)
    if next_url:
        if not next_url.startswith("http"):
            next_url = base_url + next_url
        print(f"\n Indo para a próxima página: {next_url}\n")
        url = next_url
    else:
        print("\n Nenhuma próxima página encontrada. Fim da raspagem.")
        break

driver.quit()
