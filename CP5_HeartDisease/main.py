import uvicorn
import pickle

import numpy as np
from fastapi import FastAPI, HTTPException

try:
    with open("Sprint4.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("ERRO FATAL: Arquivo do modelo 'lda_risk_dropout.pkl' não encontrado.")
    model = None
except Exception as e:
    print(f"ERRO FATAL: Não foi possível carregar o modelo. Erro: {e}")
    model = None

app = FastAPI(
    title="API de Predição de Risco de Evasão (Simples)",
    description="API para prever o risco de evasão do paciente usando parâmetros de consulta."
)

@app.get("/predict")
async def predict_risk_query(
    age,
    sex,	
    cp,	
    trestbps,
    chol,
    fbs,	
    restecg,	
    thalach,	
    exang,	
    oldpeak,	
    slope,	
    ca,	
    thal
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não está carregado. Por favor, verifique os logs do servidor."
        )

    try:
        feature_list = [
            age,
            sex,	
            cp,	
            trestbps,
            chol,
            fbs,	
            restecg,	
            thalach,	
            exang,	
            oldpeak,	
            slope,	
            ca,	
            thal
        ]

        input_data = np.array(feature_list, dtype=np.float64)

        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)

        try:
            class_1_index = model.classes_.tolist().index(1)
        except ValueError:
            class_1_index = 1

        condition = int(prediction[0])
        condition_probability = float(probabilities[0][class_1_index])

        return {
            "condition": 'Não está em risco' if condition == 0 else 'Em risco',
            "at_risk_probability": condition_probability
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erro durante a predição: {str(e)}"
        )

@app.get("/")
def read_root():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "_main_":
    uvicorn.run(app, host="0.0.0.0", port=8000)