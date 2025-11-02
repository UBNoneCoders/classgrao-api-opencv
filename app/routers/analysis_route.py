from fastapi import APIRouter
from app.utils.state_utils import get_running, set_running
from app.controller.analysis_controller import analysis

import asyncio

# router = APIRouter(prefix="/classify", tags=["Classification"])
router = APIRouter(prefix='/analysis', tags=["Analysis"])

@router.post("/trigger")
async def trigger():
    if get_running() == False:
        set_running(True)
        asyncio.create_task(asyncio.to_thread(analysis))
        
        return {"mensagem": "Análise iniciada!"}
    else:
        return {"mensagem": "Análise já está em execução."}