import google.generativeai as genai

from app.core.config import Settings

settings = Settings.from_env()

if not settings.gemini_api_key:
    raise SystemExit("GEMINI_API_KEY is not set")

genai.configure(api_key=settings.gemini_api_key)
for model in genai.list_models():
    print(model.name)
