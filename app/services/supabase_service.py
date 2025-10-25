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


async def download_image_from_bucket(image_path: str):
    file_name = image_path.split("/")[-1]

    try:
        response = supabase.storage.from_(BUCKET_NAME).download(file_name)
        if response is None:
            return None

        file_bytes = np.asarray(bytearray(response), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


async def fetch_pending_classifications():
    try:
        response = (
            supabase.table("classifications")
            .select("*")
            .eq("has_classificated", False)
            .execute()
        )
        data = response.data or []

        results = []
        for item in data:
            image_path = item.get("image_path")
            if not image_path:
                continue

            image = await download_image_from_bucket(image_path)
            if image is None:
                print(f"Unable to download {image_path}")
                continue

            results.append(
                {"id": item.get("id"), "image_path": image_path, "image": image}
            )

        return results

    except Exception as e:
        print(f"Error fetching classifications: {e}")
        return []
