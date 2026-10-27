"""Application configuration."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App settings from environment variables."""

    # Model configuration
    MODEL_NAME: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Public model, no auth needed
    MODEL_DEVICE: str = "cpu"  # "cuda", "cpu", or "auto"
    MAX_SEQUENCE_LENGTH: int = 512
    
    # API configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Database configuration
    DATABASE_URL: str = "sqlite:///./experiments.db"
    
    # Experiment configuration
    CAPTURE_ATTENTION: bool = True
    CAPTURE_HIDDEN_STATES: bool = True
    CAPTURE_MLP_ACTIVATIONS: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
