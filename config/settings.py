from __future__ import annotations
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    Pydantic-settings v2 rule:
      - The attribute name IS the env var name (case-insensitive by default).
      - No alias needed — just name the field to match the env var.
      - extra="ignore" silently drops any env vars not declared here
        (prevents ValidationError for things like MLFLOW_TRACKING_USERNAME
        which are in .env but not needed as typed fields).
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # UPSTASH_KAFKA_TOPIC == upstash_kafka_topic
        extra="ignore",         # ignore unknown env vars — no ValidationError
    )

    # ── App ───────────────────────────────────────────────────────────────
    app_env:    str = Field("development",            alias="APP_ENV")
    secret_key: str = Field("dev-secret-change-me",   alias="SECRET_KEY")
    log_level:  str = Field("INFO",                   alias="LOG_LEVEL")
    model_path: str = Field("models/xgboost_v1.json", alias="MODEL_PATH")

    # ── Upstash Kafka ─────────────────────────────────────────────────────
    upstash_kafka_bootstrap_servers: str = Field("", alias="UPSTASH_KAFKA_BOOTSTRAP_SERVERS")
    upstash_kafka_username:          str = Field("", alias="UPSTASH_KAFKA_USERNAME")
    upstash_kafka_password:          str = Field("", alias="UPSTASH_KAFKA_PASSWORD")
    upstash_kafka_topic:             str = Field("compliance-transactions", alias="UPSTASH_KAFKA_TOPIC")

    # ── Upstash Redis ─────────────────────────────────────────────────────
    upstash_redis_url:   str = Field("", alias="UPSTASH_REDIS_REST_URL")
    upstash_redis_token: str = Field("", alias="UPSTASH_REDIS_REST_TOKEN")

    # ── Neon Postgres ─────────────────────────────────────────────────────
    database_url:      str = Field("", alias="DATABASE_URL")
    database_url_sync: str = Field("", alias="DATABASE_URL_SYNC")

    # ── LLMs ─────────────────────────────────────────────────────────────
    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")
    groq_api_key:   str = Field("", alias="GROQ_API_KEY")

    # ── MCP ───────────────────────────────────────────────────────────────
    mcp_server_url: str = Field("http://localhost:8001", alias="MCP_SERVER_URL")

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field("", alias="MLFLOW_TRACKING_URI")

    # ── Convenience properties ────────────────────────────────────────────
    # Old code referenced settings.redis_url / settings.redis_token —
    # keep these as properties so nothing else breaks.

    @property
    def redis_url(self) -> str:
        return self.upstash_redis_url

    @property
    def redis_token(self) -> str:
        return self.upstash_redis_token

    @property
    def kafka_bootstrap_servers(self) -> str:
        return self.upstash_kafka_bootstrap_servers

    @property
    def kafka_username(self) -> str:
        return self.upstash_kafka_username

    @property
    def kafka_password(self) -> str:
        return self.upstash_kafka_password

    @property
    def kafka_topic(self) -> str:
        return self.upstash_kafka_topic


settings = Settings()
