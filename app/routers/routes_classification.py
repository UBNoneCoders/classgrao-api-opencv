from fastapi import APIRouter, UploadFile, File
from app.services.classification_service import processar_imagem, processar_imagem1

router = APIRouter(prefix="/classificacao", tags=["Classificação"])

@router.post("/")
async def analisar(file: UploadFile = File(...)):
    return await processar_imagem1(file)
