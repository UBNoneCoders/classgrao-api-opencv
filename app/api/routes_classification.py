from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.services.classification_service import processar_imagem

router = APIRouter(prefix="/classificar", tags=["Classificação"])

@router.post("/")
async def classificar(file: UploadFile = File(...)):
    try:
        resultado = await processar_imagem(file)
        return resultado
    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})
