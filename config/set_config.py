import logging
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

log = logging.getLogger(__name__)

class Project(BaseModel):
    name: str
    version: str
    active: str


class Redis(BaseModel):
    url: str
    ttl: int


class Milvus(BaseModel):
    host: str
    port: int
    collection: str


class Postgres(BaseModel):
    url: str


class Model(BaseModel):
    api_key: str = None
    model_name: str
    url: str = None

class Models(BaseModel):
    llm: Model
    embedding: Model
    reranker: Model


class ContextManager(BaseModel):
    max_context_tokens: int
    max_history_length: int
    compression_threshold: int

class Settings(BaseSettings):
    project: Project
    redis: Redis
    milvus: Milvus
    postgres: Postgres
    models: Models
    context_manager: ContextManager


def load_configs():
    base_path = Path(__file__).parents[1]

    main_path = base_path / "application.yaml"

    with open(main_path, "r", encoding="utf-8") as f:
        main_config = yaml.safe_load(f)

    active_env = main_config["project"]["active"]
    active_path = base_path / f"application-{active_env}.yaml"

    with open(active_path, "r", encoding="utf-8") as f:
        active_config = yaml.safe_load(f)

    all_configs = main_config | active_config

    settings = Settings(**all_configs)

    return settings

