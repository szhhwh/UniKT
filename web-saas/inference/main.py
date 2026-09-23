"""UniKT SaaS 推理服务（FastAPI，端口 8100）。

职责边界：只做模型推理相关端点；用户/会话/编排一律在 SpringBoot 后端。
SpringBoot 通过 http://localhost:8100/{health,models,predict} 调用本服务。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from engine import engine

app = FastAPI(title="UniKT Inference Service", version="0.1.0")


class PredictRequest(BaseModel):
    model: Annotated[str, Field(min_length=1)]
    questions: Annotated[list[int], Field(min_length=2)]
    skills: Annotated[list[int], Field(min_length=2)]
    responses: Annotated[list[int], Field(min_length=2)]

    @model_validator(mode="after")
    def _check_aligned(self) -> "PredictRequest":
        if not (len(self.questions) == len(self.skills) == len(self.responses)):
            raise ValueError("questions/skills/responses 长度必须一致")
        if any(r not in (0, 1) for r in self.responses):
            raise ValueError("responses 取值只能是 0/1")
        return self


class PredictResponse(BaseModel):
    model: str
    predictions: list[float]


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "unikt-inference",
        "modelCount": len(engine.available_models),
    }


@app.get("/models")
def models() -> list[dict[str, object]]:
    return [
        {"name": name, "available": info["available"], "numSkills": info["numSkills"]}
        for name, info in engine.available_models.items()
    ]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    known = engine.available_models
    if req.model not in known:
        raise HTTPException(status_code=404, detail=f"未知模型：{req.model}")
    try:
        preds = engine.predict(
            req.model, req.questions, req.skills, req.responses
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return PredictResponse(model=req.model, predictions=preds)
