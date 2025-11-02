from supabase import create_client
import numpy as np
import cv2
import os
from dotenv import load_dotenv


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
BUCKET_NAME = "classification-images"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def download_image(image_path: str):
    file_name = image_path.split("/")[-1]

    try:
        response = supabase.storage.from_(BUCKET_NAME).download(file_name)
        if response is None:
            return None

        file_bytes = np.asarray(bytearray(response), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        return None


def fetch_pending_classification():
    try:
        response = (
            supabase.table("classifications")
            .select("id, image_path")
            .eq("has_classificated", False)
            .limit(1)
            .execute()
        ).data

        if response == None:
            return None
        
        data = response[0]
        image = download_image(data['image_path'])

        if image is None:

            return None

        return data | {'image': image}

    except Exception as e:
        return None


def update_classification(id: str, analysis: dict) -> None:
    try:
        supabase.table("classifications").update(
            {"has_classificated": True, "result": analysis}
        ).eq("id", id).execute()
    except Exception:
        pass