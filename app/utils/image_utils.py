import cv2
import numpy as np
import polars as pl
from app.services.supabase_service import upload_result_image


def read_image(file):
    contents = file.read()
    nparr = np.frombuffer(contents, np.uint8)
    imagem = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return imagem


def analysis_image(file):
    """
    Analisa uma imagem para identificar e classificar grãos com base em área,
    circularidade e cor. Utiliza operações morfológicas, transformada de distância
    e rotulagem de componentes conectados para segmentar os objetos.

    Args:
        file (numpy.ndarray): Imagem de entrada em formato BGR (OpenCV).

    Returns:
        tuple:
            - result_image_path (str): Caminho da imagem resultante marcada.
            - result (dict): Dicionário contendo métricas da análise:
                * total_grains (int): Total de grãos detectados.
                * good_grains (int): Número de grãos classificados como bons.
                * bad_grains (int): Número de grãos classificados como ruins.
                * good_grains_percentage (float): Percentual de grãos bons.
                * average_area (float): Área média dos grãos.
                * average_circularity (float): Circularidade média.
                * average_color (tuple): Média das cores (R, G, B).
    """

    # Redimensiona para tamanho padrão (960x1280)
    img = cv2.resize(file, (960, 1280))

    # Converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplica desfoque Gaussiano para reduzir ruído
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # Binariza com Otsu
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Elemento estruturante para morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    # Erosão para reduzir pequenos ruídos
    eroded = cv2.erode(thresh, kernel, iterations=1)

    # Transformada de distância para identificar centros dos objetos
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)

    # Normaliza transformada de distância para 0–1
    dist_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    # Dilata para estimar o fundo seguro
    sure_bg = cv2.dilate(eroded, kernel, iterations=3)

    # Threshold para extrair o primeiro plano seguro
    _, sure_fg = cv2.threshold(dist_norm, 0.5 * dist_norm.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Área desconhecida (fronteira)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Rotulagem dos componentes conectados no primeiro plano
    _, markers = cv2.connectedComponents(sure_fg)

    # Critérios de classificação por área e cor
    min_seed_area = 100
    max_seed_area = 350
    min_color = (221, 184, 2)
    max_color = (254, 251, 141)

    results = []
    img_marked = img.copy()

    # Itera pelos marcadores (ignora 0 e 1 que são fundo)
    for marker_id in range(2, markers.max() + 1):
        # Máscara do grão atual
        mask = np.uint8(markers == marker_id)
        
        # Encontra contorno do grão
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        
        # Calcula área e perímetro
        area = cv2.contourArea(contour)
        perimetro = cv2.arcLength(contour, True)

        # Circularidade = 4π(area / perímetro²)
        if perimetro != 0:
            circularity = 4 * np.pi * (area / (perimetro ** 2))
        else:
            circularity = 0
        
        # Calcula centro do grão
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Cor média do grão
        color = cv2.mean(img, mask=mask)[:3]
        color = tuple([int(color) for color in color][::-1])  # Inverte para RGB

        # Classificação inicial
        status = 'good'

        # Verifica área permitida
        if min_seed_area > area or area > max_seed_area:
            status = 'bad'

        # Verifica intervalo de cor permitido
        if min_color > color or color > max_color:
            status = 'bad'
        
        # Marca grão na imagem
        if status == 'good':
            cv2.circle(img_marked, (cX, cY), 5, (255, 0, 0), -1)  # Azul = bom
        else:
            cv2.circle(img_marked, (cX, cY), 5, (0, 0, 255), -1)  # Vermelho = ruim
        
        # Número do grão
        cv2.putText(img_marked, str(marker_id - 1), (cX + 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Salva métricas
        results.append({
            'area': area,
            'circularity': circularity,
            'color': color,
            'status': status,
        })

    # Salva imagem marcada
    result_image_path = upload_result_image(img_marked)

    # Converte lista em DataFrame para cálculo das métricas
    df = pl.DataFrame(results)

    result = {
        "total_grains": len(df),
        "good_grains": len(df.filter(pl.col('status') == 'good')),
        "bad_grains": len(df.filter(pl.col('status') == 'bad')),
        "good_grains_percentage": round(len(df.filter(pl.col('status') == 'good')) / len(df) * 100, 2),
        "average_area": round(df['area'].mean(), 2),
        "average_circularity": round(df['circularity'].mean(), 2),
        "average_color": (
            int(df['color'].list.get(0).mean()),
            int(df['color'].list.get(1).mean()),
            int(df['color'].list.get(2).mean())
        ),
    }

    return result_image_path, result