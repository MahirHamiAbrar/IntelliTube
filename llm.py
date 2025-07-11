from typing import Literal, Optional
from langchain.chat_models import init_chat_model as __init_chat_model
from langchain_core.language_models import BaseChatModel

from dotenv import load_dotenv
load_dotenv()

MODEL_PROVIDERS = {
        'openai': 'openai',
        'groq': 'groq',
        'nvidia': 'nvidia',
        'google': 'google_genai',
        'ollama': 'ollama',
    }

DEFAULT_MODELS = {
    'openai': "gpt-4o-mini",
    'groq': "llama-3.3-70b-versatile",
    'nvidia': "mistralai/mistral-small-24b-instruct",
    'google': "gemini-2.0-flash",
    'ollama': "llama3.2:3b",
}

def init_llm(
    model_provider: Literal['openai', 'groq', 'nvidia', 'google', 'ollama'],
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    global DEFAULT_MODELS, MODEL_PROVIDERS
    
    params = {
        "model": model_name or DEFAULT_MODELS.get(model_provider),
        "model_provider": MODEL_PROVIDERS.get(model_provider) or model_provider,
        "temperature": temperature
    }

    if api_key:
        params["api_key"] = api_key

    return __init_chat_model(**params, **kwargs)
