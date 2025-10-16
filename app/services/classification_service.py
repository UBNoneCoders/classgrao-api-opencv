import cv2
import numpy as np
from app.utils.image_utils import ler_imagem

async def processar_imagem(file):
    imagem = await ler_imagem(file)

    if imagem is None:
        return {"erro": "Imagem inválida"}

    # Exemplo: extrair média de cor (placeholder para futura classificação)
    media_cor = cv2.mean(imagem)[:3]  # B, G, R

    resultado = {
        "mensagem": "Imagem processada com sucesso!",
        "media_cor_BGR": [round(c, 2) for c in media_cor]
    }

    return resultado
