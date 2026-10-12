"""naptrap: local, text-defined wake word and keyword detection."""

from .artifacts import ArtifactError
from .audio import load_wav
from .google import GoogleSpeechEmbedding
from .models import PhonMatchNet
from .registry import available_models, load_model, register_model
from .text import G2PTextEncoder, normalize_keyword
from .types import Keyword, KeywordModel, Prediction

__version__ = "0.1.0"
__all__ = [
    "ArtifactError", "G2PTextEncoder", "GoogleSpeechEmbedding", "Keyword", "KeywordModel", "PhonMatchNet",
    "Prediction", "available_models", "load_model", "load_wav", "normalize_keyword", "register_model",
]
