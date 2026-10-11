"""Naptrap: local inference for text-defined keyword models."""

from .bundle import BundleError
from .google import GoogleSpeechEmbedding
from .models import PhonMatchNet
from .registry import available_models, load_model, register_model
from .types import KeywordModel, Prediction

__version__ = "0.1.0"
__all__ = ["BundleError", "GoogleSpeechEmbedding", "KeywordModel", "PhonMatchNet", "Prediction", "available_models", "load_model", "register_model"]
