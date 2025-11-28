import os
from fastapi import APIRouter, HTTPException
from app.utils.state_utils import get_running, set_running
from app.controller.analysis_controller import analysis
from pydantic import BaseModel

import asyncio

TRIGGER_PASSWORD = os.getenv("TRIGGER_PASSWORD")

class TriggerRequest(BaseModel):
    password: str

router = APIRouter(prefix='/analysis', tags=["Analysis"])

@router.post("/trigger")
async def trigger(payload: TriggerRequest):
    if payload.password != TRIGGER_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized: wrong password")

    if get_running() == False:
        set_running(True)
        asyncio.create_task(asyncio.to_thread(analysis))
        
        return {"mensagem": "Análise iniciada!"}
    else:
        return {"mensagem": "Análise já está em execução."}