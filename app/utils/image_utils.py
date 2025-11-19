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
    img = cv2.resize(file, (960, 1280))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    eroded = cv2.erode(thresh, kernel, iterations=1)

    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    dist_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    sure_bg = cv2.dilate(eroded, kernel, iterations=3)
    _, sure_fg = cv2.threshold(dist_norm, 0.5 * dist_norm.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)

    min_seed_area = 100
    max_seed_area = 350
    min_color = (221, 184, 2)
    max_color = (254, 251, 141)

    results = []
    img_marked = img.copy()
    for marker_id in range(2, markers.max() + 1):
        mask = np.uint8(markers == marker_id)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        
        area = cv2.contourArea(contour)
        perimetro = cv2.arcLength(contour, True)
        if perimetro != 0:
            circularity = 4 * np.pi * (area / (perimetro ** 2))
        else:
            circularity = 0
        
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        color = cv2.mean(img, mask=mask)[:3]
        color = tuple([int(color) for color in color][::-1])

        status = 'good'

        if min_seed_area > area or area > max_seed_area:
            status = 'bad'
        if min_color > color or color > max_color:
            status = 'bad'
        

        if status == 'good':
            cv2.circle(img_marked, (cX, cY), 5, (255, 0, 0), -1)
        else:
            cv2.circle(img_marked, (cX, cY), 5, (0, 0, 255), -1)
        
        cv2.putText(img_marked, str(marker_id - 1), (cX + 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        results.append({
            'area': area,
            'circularity': circularity,
            'color': color,
            'status': status,
        })

    result_image_path = upload_result_image(img_marked)
    df = pl.DataFrame(results)
    result = {
        "total_grains": len(df),
        "good_grains": len(df.filter(pl.col('status') == 'good')),
        "bad_grains": len(df.filter(pl.col('status') == 'bad')),
        "good_grains_percentage": round(len(df.filter(pl.col('status') == 'good')) / len(df) * 100, 2),
        "average_area": round(df['area'].mean(), 2),
        "average_circularity": round(df['circularity'].mean(), 2),
        "average_color": (int(df['color'].list.get(0).mean()), int(df['color'].list.get(1).mean()), int(df['color'].list.get(2).mean())),
    }

    return result_image_path, result