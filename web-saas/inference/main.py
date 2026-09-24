"""UniKT SaaS 推理服务（FastAPI，端口 8100）.

职责边界：只做模型推理相关端点；用户/会话/编排一律在 SpringBoot 后端。
SpringBoot 通过 http://localhost:8100/{health,models,skills,predict} 调用本服务。

部署在可达网络（如 Tailscale IP）时，设置环境变量 UNIKT_INFERENCE_TOKEN：
所有请求必须携带一致的 X-Inference-Token 头，否则 401——防止绕过门户
的鉴权与配额直连推理。
"""

from __future__ import annotations

import os
from typing import Annotated

from engine import engine
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

app = FastAPI(title="UniKT Inference Service", version="0.1.0")

_INFERENCE_TOKEN = os.environ.get("UNIKT_INFERENCE_TOKEN", "")


@app.middleware("http")
async def _token_guard(request: Request, call_next):
    """共享密钥校验：设置了 UNIKT_INFERENCE_TOKEN 后所有路径都要求该头."""
    if _INFERENCE_TOKEN and request.headers.get("X-Inference-Token") != _INFERENCE_TOKEN:
        return JSONResponse(
            {"detail": "推理服务需要 X-Inference-Token（经门户后端调用）"}, status_code=401
        )
    return await call_next(request)


class PredictRequest(BaseModel):
    """一次预测请求：一条学习者作答序列.

    questions 为可选的兼容字段（与 skills 同义的旧占位），新调用方
    只需 skills + responses。
    """

    model: Annotated[str, Field(min_length=1)]
    questions: Annotated[list[int] | None, Field(min_length=2)] = None
    skills: Annotated[list[int], Field(min_length=2)]
    responses: Annotated[list[int], Field(min_length=2)]

    @model_validator(mode="after")
    def _check_aligned(self) -> PredictRequest:
        """校验序列等长且 responses 只含 0/1."""
        if self.questions is not None and len(self.questions) != len(self.skills):
            raise ValueError("questions/skills 长度必须一致")
        if not (len(self.skills) == len(self.responses)):
            raise ValueError("skills/responses 长度必须一致")
        if any(r not in (0, 1) for r in self.responses):
            raise ValueError("responses 取值只能是 0/1")
        return self


class PredictResponse(BaseModel):
    """预测结果：逐步掌握度概率列表."""

    model: str
    predictions: list[float]


@app.get("/health")
def health() -> dict[str, object]:
    """存活探测 + 已注册模型数."""
    return {
        "status": "ok",
        "service": "unikt-inference",
        "modelCount": len(engine.available_models),
    }


@app.get("/models")
def models() -> list[dict[str, object]]:
    """模型清单：注册名 + 是否已训练 + 数据集知识点数."""
    return [
        {"name": name, "available": info["available"], "numSkills": info["numSkills"]}
        for name, info in engine.available_models.items()
    ]


@app.get("/skills/{model}")
def skills(model: str) -> list[dict[str, object]]:
    """该模型训练数据的技能目录（id/名称/关联题量）；未训练返回 404."""
    catalog = engine.skill_catalog(model)
    if catalog is None:
        raise HTTPException(status_code=404, detail=f"模型 {model} 尚未训练，无技能目录")
    return catalog


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """对作答序列做真实模型推理，返回逐步答对概率."""
    known = engine.available_models
    if req.model not in known:
        raise HTTPException(status_code=404, detail=f"未知模型：{req.model}")
    try:
        preds = engine.predict(req.model, req.questions, req.skills, req.responses)
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return PredictResponse(model=req.model, predictions=preds)
