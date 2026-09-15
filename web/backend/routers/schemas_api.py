"""Schemas API router — model discovery and parameter schema retrieval.

Provides endpoints to list available model names and retrieve the
parameter schema (groups, fields, defaults) for a specific model.
"""

from typing import Any

from dependencies import get_schema_extractor
from errors import AppError
from fastapi import APIRouter
from schemas import ModelSchemaResponse, ParamGroup
from services.schema_extractor import SchemaExtractor

router = APIRouter(prefix="/api/schemas", tags=["schemas"])


def _get_extractor() -> SchemaExtractor:
    """Return the shared SchemaExtractor singleton.

    Returns:
        The application-wide SchemaExtractor instance.
    """
    return get_schema_extractor()


@router.get("/models", response_model=list[str])
def list_models() -> Any:
    """List all available model names.

    Returns:
        A list of model name strings.
    """
    return _get_extractor().list_models()


@router.get("/models/{model_name}/params", response_model=ModelSchemaResponse)
def get_model_params(model_name: str) -> Any:
    """Return the parameter schema for a specific model.

    Args:
        model_name: The model name to look up.

    Returns:
        A ModelSchemaResponse with parameter groups and fields.

    Raises:
        AppError: 404 if the model is not found.
    """
    try:
        return _get_extractor().get_model_schema(model_name)
    except KeyError:
        raise AppError("model_not_found", 404)


@router.get("/preprocess/{action}", response_model=list[ParamGroup])
def get_preprocess_params(action: str) -> Any:
    """Return the parameter schema for a preprocess action (download/process).

    Args:
        action: Preprocess action ('download' or 'process').

    Returns:
        A list of ParamGroup describing the action's parameters.

    Raises:
        AppError: 404 if the action is unknown.
    """
    try:
        return _get_extractor().get_preprocess_schema(action)
    except KeyError:
        raise AppError("preprocess_action_unknown", 404)
