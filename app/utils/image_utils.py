import cv2
import numpy as np

def read_image(file):
    contents = file.read()
    nparr = np.frombuffer(contents, np.uint8)
    imagem = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return imagem


def analysis_image(file):
    # image = read_image(file)
    image = cv2.resize(file , (800, 800))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    circularities = []
    impurities = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0 or area < 20:
            continue

        circularity = 4 * np.pi * (area / (perimeter**2))
        areas.append(area)
        circularities.append(circularity)

        if area < 300 or circularity < 0.6:
            impurities.append(contour)

    total_count = len(contours)
    impurities_count = len(impurities)
    impurities_percentage = (
        (impurities_count / total_count * 100) if total_count > 0 else 0
    )
    average_color = cv2.mean(image)[:3]

    result = {
        "total_grains": total_count,
        "total_impurities": impurities_count,
        "impurities_percentage": round(impurities_percentage, 2),
        "average_area": round(np.mean(areas), 2) if areas else 0,
        "average_circularity": round(np.mean(circularities), 3) if circularities else 0,
        "average_color_bgr": [round(c, 2) for c in average_color],
    }

    return result