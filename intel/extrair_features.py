import os
import cv2
import numpy as np
import pandas as pd

PASTA_IMAGENS = "data/imagens"

def extrair_features(caminho_imagem):
    img = cv2.imread(caminho_imagem)
    if img is None:
        return None


    media_cor = cv2.mean(img)[:3]
    std_cor = np.std(img, axis=(0, 1))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contornos]
    perimetros = [cv2.arcLength(c, True) for c in contornos]
    circularidades = [
        (4 * np.pi * a) / (p ** 2 + 1e-6)
        for a, p in zip(areas, perimetros) if p > 0
    ]

    area_media = np.mean(areas) if areas else 0
    circularidade_media = np.mean(circularidades) if circularidades else 0
    qtd_graos = len(contornos)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    textura_media = np.mean(np.abs(laplacian))
    textura_std = np.std(laplacian)

    features = {
        "media_B": media_cor[0],
        "media_G": media_cor[1],
        "media_R": media_cor[2],
        "std_B": std_cor[0],
        "std_G": std_cor[1],
        "std_R": std_cor[2],
        "area_media": area_media,
        "circularidade_media": circularidade_media,
        "qtd_graos": qtd_graos,
        "textura_media": textura_media,
        "textura_std": textura_std,
    }

    return features

dados = []
for nome_arquivo in os.listdir(PASTA_IMAGENS):
    caminho = os.path.join(PASTA_IMAGENS, nome_arquivo)
    features = extrair_features(caminho)
    if features:
        rotulo = nome_arquivo.split("_")[0]  
        features["rotulo"] = rotulo
        features["arquivo"] = nome_arquivo
        dados.append(features)

df = pd.DataFrame(dados)
df.to_csv("data/features.csv", index=False)
print(" Extração concluída! Arquivo salvo em data/features.csv")
