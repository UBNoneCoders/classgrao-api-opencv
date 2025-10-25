from fastapi import APIRouter, HTTPException
from app.services.supabase_service import fetch_pending_classifications
from app.services.classification_service import process_image_basic
from app.services.supabase_service import supabase
from pydantic import BaseModel
import os

router = APIRouter(prefix="/classify", tags=["Classification"])

TRIGGER_PASSWORD = os.getenv("TRIGGER_PASSWORD")

class TriggerRequest(BaseModel):
    password: str

@router.post("/trigger")
async def trigger_analysis(payload: TriggerRequest):
    if payload.password != TRIGGER_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized: wrong password")

    pending_items = await fetch_pending_classifications()

    if not pending_items:
        return {"message": "No pending classifications found."}

    results = []
    for item in pending_items:
        result = await process_image_basic(item["image"])
        results.append(
            {"id": item["id"], "image_path": item["image_path"], "result": result}
        )

        try:
            supabase.table("classifications").update(
                {"has_classificated": True, "result": result}
            ).eq("id", item["id"]).execute()
        except Exception:
            return {
                "message": f"Processed {len(results)} images.",
                "results": results,
            }

    return {
        "message": f"Processed {len(results)} images.",
        "results": results,
    }
