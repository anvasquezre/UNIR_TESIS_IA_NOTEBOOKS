from typing import Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    OPENAI_API_KEY: Optional[SecretStr] = None
    NPY_ROOT_PATH: str = "./mpdocvqa_imdbs"


settings = Settings()
