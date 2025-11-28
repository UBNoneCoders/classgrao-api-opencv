from fastapi import FastAPI
from app.routers.analysis_route import router

app = FastAPI(title="API de Classificação de Grãos", version="1.0")

app.include_router(router)

@app.get("/")
def root():
    return {"mensagem": "API de Classificação de Grãos está no ar"}
