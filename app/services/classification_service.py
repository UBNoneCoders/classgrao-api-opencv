# app/services/classificacao_service.py
import cv2
import numpy as np
from app.utils.image_utils import ler_imagem


async def process_image_basic(file_or_array):
    if isinstance(file_or_array, np.ndarray):
        image = file_or_array
    else:
        image = await ler_imagem(file_or_array)

    if image is None:
        return {"erro": "Imagem inválida"}

    image = cv2.resize(image, (800, 800))
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

    output_image = image.copy()
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 1)
    cv2.imwrite("resultado_debug.jpg", output_image)

    return result


async def process_image_advanced(file, debug_save_path="resultado_debug.jpg"):
    image = await ler_imagem(file)
    height, width = image.shape[:2]
    max_dimension = 1200

    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    background_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    background = cv2.morphologyEx(blur, cv2.MORPH_OPEN, background_kernel)
    normalized = cv2.subtract(blur, background)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

    thresh = cv2.adaptiveThreshold(
        normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2
    )

    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        opened, 8, cv2.CV_32S
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = max(50, 0.0005 * (image.shape[0] * image.shape[1]))
    mask = np.zeros_like(opened)

    for i, area in enumerate(areas, start=1):
        if area >= min_area:
            mask[labels == i] = 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_contours = []
    if len(contours) > 0:
        mask_color = np.zeros_like(gray)
        cv2.drawContours(mask_color, contours, -1, 255, -1)
        distance = cv2.distanceTransform(mask_color, cv2.DIST_L2, 5)
        _, sure_foreground = cv2.threshold(distance, 0.4 * distance.max(), 255, 0)
        sure_foreground = np.uint8(sure_foreground)
        unknown = cv2.subtract(mask_color, sure_foreground)

        num_markers, markers = cv2.connectedComponents(sure_foreground)
        markers = markers + 1
        markers[unknown == 255] = 0
        watershed_image = image.copy()
        cv2.watershed(watershed_image, markers)

        for marker_id in range(2, num_markers + 2):
            marker_mask = np.uint8(markers == marker_id)
            marker_contours, _ = cv2.findContours(
                marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in marker_contours:
                if cv2.contourArea(contour) > min_area:
                    final_contours.append(contour)

    if not final_contours:
        final_contours = contours

    areas_list = []
    circularities_list = []
    impurities = []

    for contour in final_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        areas_list.append(area)
        circularities_list.append(circularity)

        if area < (2 * min_area) or circularity < 0.5:
            impurities.append(contour)

    total_count = len(final_contours)
    impurities_count = len(impurities)
    impurities_percentage = (
        (impurities_count / total_count * 100) if total_count > 0 else 0
    )
    global_average_color = cv2.mean(image)[:3]

    debug_image = image.copy()
    cv2.drawContours(debug_image, final_contours, -1, (0, 255, 0), 1)
    cv2.drawContours(debug_image, impurities, -1, (0, 0, 255), 2)
    cv2.imwrite(debug_save_path, debug_image)

    result = {
        "total_grains": int(total_count),
        "total_impurities": int(impurities_count),
        "impurities_percentage": round(impurities_percentage, 2),
        "average_area": float(np.mean(areas_list)) if areas_list else 0.0,
        "average_circularity": (
            float(np.mean(circularities_list)) if circularities_list else 0.0
        ),
        "average_color_bgr": [float(round(c, 2)) for c in global_average_color],
    }

    return result
