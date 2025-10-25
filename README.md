# Classificador de Grãos

API em Python para análise e classificação de grãos usando OpenCV, com FastAPI.  
O projeto permite pré-processar imagens de amostras, segmentar grãos e impurezas, extrair características (área, circularidade, cor média) e retornar um JSON com resultados quantitativos e estatísticos da amostra.

---

## 🔹 Funcionalidades

- Pré-processamento de imagens (cinza, blur, equalização adaptativa).
- Segmentação de grãos e detecção de impurezas.
- Extração de características simples: área, circularidade, cor média.
- Cálculo do percentual de impurezas na amostra.
- API RESTful com FastAPI para envio de imagens e retorno de resultados em JSON.
- Estrutura modular, permitindo fácil integração com modelos de Machine Learning futuros.

---

## 📂 Estrutura do projeto

```

classificador_graos/
├── app/
│   ├── main.py                      # Ponto de entrada da API
│   ├── core/
│   │   └── config.py                # Configurações globais
│   ├── routers/
│   │   └── routes_classification.py # Endpoints da API
│   ├── services/
│   │   └── classificacao_service.py # Lógica de análise e processamento
│   ├── utils/
│   │   └── image_utils.py           # Funções auxiliares (ler/converter imagem)
│   └── models/
│       └── schemas.py               # Schemas Pydantic de entrada/saída
└── intel/
├── scrape_images.py             # Script de coleta de imagens
└── extrair_features.py          # Extração de features para treino ML

````

---

## ⚙️ Instalação

1. Clone o repositório:
```bash
git clone https://github.com/UBNoneCoders/classgrao-api-opencv.git
cd classificador_graos
````

2. Crie e ative um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

> **Dependências principais:** `fastapi`, `uvicorn`, `opencv-python`, `numpy`.

---

## 🚀 Executando a API

```bash
uvicorn app.main:app --reload
```

Acesse no navegador:

```
http://127.0.0.1:8000/docs
```

para abrir a interface interativa do Swagger.

## 👥 Integrantes
- [Guilherme Felipe Mendonça](https://github.com/guilherme-felipe123)
- [Matheus Augusto Silva dos Santos](https://github.com/Matheuz233)
- [Luan Jacomini Klho](https://github.com/luanklo)


