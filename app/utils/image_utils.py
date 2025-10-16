import cv2
import numpy as np

async def ler_imagem(file):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    imagem = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return imagem
