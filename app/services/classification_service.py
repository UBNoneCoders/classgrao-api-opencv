# app/services/classificacao_service.py
import cv2
import numpy as np
from app.utils.image_utils import ler_imagem

async def processar_imagem1(file):
    imagem = await ler_imagem(file)

    if imagem is None:
        return {"erro": "Imagem inválida"}


    imagem = cv2.resize(imagem, (800, 800))
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)


    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    circularidades = []
    impurezas = []

    for c in contours:
        area = cv2.contourArea(c)
        perimetro = cv2.arcLength(c, True)
        if perimetro == 0 or area < 20:
            continue

        circularidade = 4 * np.pi * (area / (perimetro ** 2))
        areas.append(area)
        circularidades.append(circularidade)

        if area < 300 or circularidade < 0.6:
            impurezas.append(c)

    qtd_total = len(contours)
    qtd_impurezas = len(impurezas)
    percentual_impurezas = (qtd_impurezas / qtd_total * 100) if qtd_total > 0 else 0

    media_cor = cv2.mean(imagem)[:3]  # B, G, R

    resultado = {
        "mensagem": "Análise concluída com sucesso!",
        "qtd_total_graos": qtd_total,
        "qtd_impurezas": qtd_impurezas,
        "percentual_impurezas": round(percentual_impurezas, 2),
        "area_media": round(np.mean(areas), 2) if areas else 0,
        "circularidade_media": round(np.mean(circularidades), 3) if circularidades else 0,
        "media_cor_BGR": [round(c, 2) for c in media_cor],
    }
    img_out = imagem.copy()
    cv2.drawContours(img_out, contours, -1, (0,255,0), 1)
    cv2.imwrite("resultado_debug.jpg", img_out)

    return resultado


async def processar_imagem(file, debug_save_path="resultado_debug.jpg"):
    """
    Recebe imagem (BGR numpy array). Retorna dicionário de resultados.
    Se debug_save_path fornecido, salva imagem com marcações.
    """

    imagem = await ler_imagem(file)
    h, w = imagem.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        imagem = cv2.resize(imagem, (int(w*scale), int(h*scale)))

    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    bg = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel_bg)
    norm = cv2.subtract(blur, bg) 
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)

    thresh = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 2)

    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, 8, cv2.CV_32S)
    areas = stats[1:, cv2.CC_STAT_AREA]  
    img_area = imagem.copy()
    min_area = max(50, 0.0005 * (imagem.shape[0]*imagem.shape[1])) 
    mask = np.zeros_like(opened)
    for i, a in enumerate(areas, start=1):
        if a >= min_area:
            mask[labels == i] = 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_contours = []
    if len(contours) > 0:
        mask_color = np.zeros_like(gray)
        cv2.drawContours(mask_color, contours, -1, 255, -1)
        dist = cv2.distanceTransform(mask_color, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(mask_color, sure_fg)

        num_markers, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        img_watershed = imagem.copy()
        cv2.watershed(img_watershed, markers)

        for marker_id in range(2, num_markers+2):
            m = np.uint8(markers == marker_id)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) > min_area:
                    final_contours.append(c)

    if not final_contours:
        final_contours = contours

    areas_list = []
    circ_list = []
    impurezas = []
    for c in final_contours:
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        if p == 0:
            continue
        circ = 4 * np.pi * (a / (p * p))
        areas_list.append(a)
        circ_list.append(circ)
        if a < (2 * min_area) or circ < 0.5:
            impurezas.append(c)

    qtd_total = len(final_contours)
    qtd_impurezas = len(impurezas)
    percentual_impurezas = (qtd_impurezas / qtd_total * 100) if qtd_total > 0 else 0

    media_global = cv2.mean(imagem)[:3]

    debug_img = imagem.copy()
    cv2.drawContours(debug_img, final_contours, -1, (0,255,0), 1)
    cv2.drawContours(debug_img, impurezas, -1, (0,0,255), 2) 
    cv2.imwrite(debug_save_path, debug_img)
    

    resultado = {
        "qtd_total_graos": int(qtd_total),
        "qtd_impurezas": int(qtd_impurezas),
        "percentual_impurezas": round(percentual_impurezas, 2),
        "area_media": float(np.mean(areas_list)) if areas_list else 0.0,
        "circularidade_media": float(np.mean(circ_list)) if circ_list else 0.0,
        "media_cor_BGR": [float(round(c,2)) for c in media_global]
    }
    return resultado
