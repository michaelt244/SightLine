import google.generativeai as genai

from app.core.config import Settings
from app.core.logger import configure_logging, get_logger

settings = Settings.from_env()
configure_logging()
logger = get_logger("sightline.test")

if not settings.gemini_api_key:
    raise SystemExit("GEMINI_API_KEY is not set")

genai.configure(api_key=settings.gemini_api_key)
for model in genai.list_models():
    logger.info("Gemini model available", extra={"event": "gemini_model", "context": {"name": model.name}})
