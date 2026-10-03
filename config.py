# =========================================================
# IraAI — Local AI Configuration
# =========================================================

from __future__ import annotations

import os

from typing import (
    Any,
    Dict,
    List
)


# =========================================================
# CUDA DETECTION
# =========================================================

def _cuda_available() -> bool:
    try:
        import torch

        return bool(
            torch.cuda.is_available()
        )

    except Exception:
        return False


# =========================================================
# Base Paths
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_ROOT = os.getenv(
    "IRAAI_MODEL_ROOT",
    os.path.join(
        BASE_DIR,
        "models"
    )
)

MEMORY_FILE = os.getenv(
    "IRAAI_MEMORY_FILE",
    os.path.join(
        BASE_DIR,
        "iraai_memory.json"
    )
)


# =========================================================
# Hugging Face Account
# =========================================================

HF_USERNAME = os.getenv(
    "IRAAI_HF_USERNAME",
    "atifahmed2789"
)

HF_TOKEN = os.getenv(
    "HF_TOKEN",
    os.getenv(
        "HUGGINGFACE_HUB_TOKEN",
        ""
    )
)


# =========================================================
# Model Directories
# =========================================================

GENERAL_MODEL_PATH = os.path.join(
    MODEL_ROOT,
    "general"
)

CODER_MODEL_PATH = os.path.join(
    MODEL_ROOT,
    "coder"
)

VISION_MODEL_PATH = os.path.join(
    MODEL_ROOT,
    "vision"
)

REASONING_MODEL_PATH = os.path.join(
    MODEL_ROOT,
    "reasoning"
)

STT_MODEL_PATH = os.path.join(
    MODEL_ROOT,
    "speech_to_text"
)

TTS_MODEL_PATH = os.path.join(
    MODEL_ROOT,
    "text_to_speech"
)

MUSIC_MODEL_PATH = os.path.join(
    MODEL_ROOT,
    "music"
)

VIDEO_MODEL_PATH = os.path.join(
    MODEL_ROOT,
    "video"
)

IMAGE_MODEL_PATH = os.path.join(
    MODEL_ROOT,
    "image"
)

IMAGE_REFINER_MODEL_PATH = os.path.join(
    MODEL_ROOT,
    "image_refiner"
)

EMBEDDING_MODEL_PATH = os.path.join(
    MODEL_ROOT,
    "embedding"
)


# =========================================================
# Model Names
# =========================================================

GENERAL_MODEL_NAME = "Qwen3-8-27B"

CODER_MODEL_NAME = (
    "Qwen3-Coder-30B-A3B-Instruct"
)

VISION_MODEL_NAME = (
    "Qwen3-VL-8B-Instruct"
)

REASONING_MODEL_NAME = "DeepSeek-R1"

STT_MODEL_NAME = "Whisper-Small"

TTS_MODEL_NAME = "Kokoro-82M"

MUSIC_MODEL_NAME = "MusicGen-Small"

VIDEO_MODEL_NAME = (
    "Wan2.1-T2V-1.3B"
)

IMAGE_MODEL_NAME = (
    "Stable-Diffusion-XL-Base-1.0"
)

IMAGE_REFINER_MODEL_NAME = (
    "Stable-Diffusion-XL-Refiner-1.0"
)

EMBEDDING_MODEL_NAME = "BGE-M3"


# =========================================================
# Hugging Face Repositories
# =========================================================
#
# Each repository contains one model.
#
# Hugging Face is used only as the model source.
# IraAI does not send user prompts to Hugging Face
# for AI generation.
# =========================================================

HF_REPOSITORIES = {

    GENERAL_MODEL_NAME:
        f"{HF_USERNAME}/IraAI-Qwen3-8-27B",

    CODER_MODEL_NAME:
        f"{HF_USERNAME}/"
        "IraAI-Qwen3-Coder-30B-A3B-Instruct",

    VISION_MODEL_NAME:
        f"{HF_USERNAME}/"
        "IraAI-Qwen3-VL-8B-Instruct",

    REASONING_MODEL_NAME:
        f"{HF_USERNAME}/IraAI-DeepSeek-R1",

    STT_MODEL_NAME:
        f"{HF_USERNAME}/IraAI-Whisper-Small",

    TTS_MODEL_NAME:
        f"{HF_USERNAME}/IraAI-Kokoro-82M",

    MUSIC_MODEL_NAME:
        f"{HF_USERNAME}/IraAI-MusicGen-Small",

    VIDEO_MODEL_NAME:
        f"{HF_USERNAME}/IraAI-Wan2.1-T2V-1.3B",

    IMAGE_MODEL_NAME:
        f"{HF_USERNAME}/IraAI-SDXL-Base-1.0",

    IMAGE_REFINER_MODEL_NAME:
        f"{HF_USERNAME}/IraAI-SDXL-Refiner-1.0",

    EMBEDDING_MODEL_NAME:
        f"{HF_USERNAME}/IraAI-BGE-M3",
}


# =========================================================
# Backward Compatibility
# =========================================================

HF_MODELS = HF_REPOSITORIES


# =========================================================
# Default Model
# =========================================================

DEFAULT_MODEL = os.getenv(
    "IRAAI_DEFAULT_MODEL",
    GENERAL_MODEL_NAME
)


# =========================================================
# Model Settings
# =========================================================

AI_TEMPERATURE = float(
    os.getenv(
        "IRAAI_TEMPERATURE",
        "0.7"
    )
)

AI_MAX_NEW_TOKENS = int(
    os.getenv(
        "IRAAI_MAX_NEW_TOKENS",
        "2048"
    )
)

AI_TOP_P = float(
    os.getenv(
        "IRAAI_TOP_P",
        "0.9"
    )
)

AI_DO_SAMPLE = (
    os.getenv(
        "IRAAI_DO_SAMPLE",
        "true"
    )
    .strip()
    .lower()
    == "true"
)


# =========================================================
# Device / Inference
# =========================================================

USE_CUDA = (
    os.getenv(
        "IRAAI_USE_CUDA",
        "auto"
    )
    .strip()
    .lower()
)

if USE_CUDA == "true":
    USE_GPU = True

elif USE_CUDA == "false":
    USE_GPU = False

else:
    USE_GPU = _cuda_available()


DEVICE = (
    "cuda"
    if USE_GPU
    else "cpu"
)


# =========================================================
# Offline / Local AI
# =========================================================

OFFLINE_MODE = True

LOCAL_FILES_ONLY = True

ALLOW_REMOTE_CODE = True

TRUST_REMOTE_CODE = True

EXTERNAL_AI_API_ENABLED = False


# =========================================================
# Hugging Face Model Download / Cache
# =========================================================

HF_DOWNLOAD_ENABLED = True

HF_LOCAL_CACHE_DIR = os.path.join(
    MODEL_ROOT,
    ".hf_cache"
)

HF_REVISION = os.getenv(
    "IRAAI_HF_REVISION",
    "main"
)

HF_FORCE_DOWNLOAD = (
    os.getenv(
        "IRAAI_HF_FORCE_DOWNLOAD",
        "false"
    )
    .strip()
    .lower()
    == "true"
)

HF_RESUME_DOWNLOAD = True


# =========================================================
# Memory
# =========================================================

MEMORY_ENTRY_LIMIT = None

AUTO_DELETE_MEMORY = False

MEMORY_SEARCH_LIMIT = None


# =========================================================
# Tools
# =========================================================

WEB_TOOLS_ENABLED = True

FILE_TOOLS_ENABLED = True

ANDROID_TOOLS_ENABLED = True

IMAGE_TOOLS_ENABLED = True

VIDEO_TOOLS_ENABLED = True


# =========================================================
# Server
# =========================================================

HOST = os.getenv(
    "IRAAI_HOST",
    "0.0.0.0"
)

PORT = int(
    os.getenv(
        "PORT",
        "8000"
    )
)

DEBUG = (
    os.getenv(
        "IRAAI_DEBUG",
        "false"
    )
    .strip()
    .lower()
    == "true"
)


# =========================================================
# Request Limits
# =========================================================

MAX_CONTENT_LENGTH = int(
    os.getenv(
        "IRAAI_MAX_CONTENT_LENGTH",
        str(
            100 * 1024 * 1024
        )
    )
)


# =========================================================
# Backend API Security
# =========================================================

API_KEY_REQUIRED = (
    os.getenv(
        "IRAAI_API_KEY_REQUIRED",
        "false"
    )
    .strip()
    .lower()
    == "true"
)

IRAAI_API_KEY = os.getenv(
    "IRAAI_API_KEY",
    ""
)


# =========================================================
# Complete Model Registry
# =========================================================

LOCAL_MODELS: Dict[
    str,
    Dict[str, Any]
] = {

    # -----------------------------------------------------
    # General
    # -----------------------------------------------------

    GENERAL_MODEL_NAME: {

        "type":
            "general",

        "category":
            "chat",

        "path":
            GENERAL_MODEL_PATH,

        "repository":
            HF_REPOSITORIES[
                GENERAL_MODEL_NAME
            ],

        "huggingface":
            HF_REPOSITORIES[
                GENERAL_MODEL_NAME
            ],

        "local":
            True,

        "offline":
            True
    },


    # -----------------------------------------------------
    # Coder
    # -----------------------------------------------------

    CODER_MODEL_NAME: {

        "type":
            "coder",

        "category":
            "coding",

        "path":
            CODER_MODEL_PATH,

        "repository":
            HF_REPOSITORIES[
                CODER_MODEL_NAME
            ],

        "huggingface":
            HF_REPOSITORIES[
                CODER_MODEL_NAME
            ],

        "local":
            True,

        "offline":
            True
    },


    # -----------------------------------------------------
    # Vision
    # -----------------------------------------------------

    VISION_MODEL_NAME: {

        "type":
            "vision",

        "category":
            "vision",

        "path":
            VISION_MODEL_PATH,

        "repository":
            HF_REPOSITORIES[
                VISION_MODEL_NAME
            ],

        "huggingface":
            HF_REPOSITORIES[
                VISION_MODEL_NAME
            ],

        "local":
            True,

        "offline":
            True
    },


    # -----------------------------------------------------
    # Reasoning
    # -----------------------------------------------------

    REASONING_MODEL_NAME: {

        "type":
            "reasoning",

        "category":
            "reasoning",

        "path":
            REASONING_MODEL_PATH,

        "repository":
            HF_REPOSITORIES[
                REASONING_MODEL_NAME
            ],

        "huggingface":
            HF_REPOSITORIES[
                REASONING_MODEL_NAME
            ],

        "local":
            True,

        "offline":
            True
    },


    # -----------------------------------------------------
    # Speech To Text
    # -----------------------------------------------------

    STT_MODEL_NAME: {

        "type":
            "speech_to_text",

        "category":
            "audio",

        "path":
            STT_MODEL_PATH,

        "repository":
            HF_REPOSITORIES[
                STT_MODEL_NAME
            ],

        "huggingface":
            HF_REPOSITORIES[
                STT_MODEL_NAME
            ],

        "local":
            True,

        "offline":
            True
    },


    # -----------------------------------------------------
    # Text To Speech
    # -----------------------------------------------------

    TTS_MODEL_NAME: {

        "type":
            "text_to_speech",

        "category":
            "audio",

        "path":
            TTS_MODEL_PATH,

        "repository":
            HF_REPOSITORIES[
                TTS_MODEL_NAME
            ],

        "huggingface":
            HF_REPOSITORIES[
                TTS_MODEL_NAME
            ],

        "local":
            True,

        "offline":
            True
    },


    # -----------------------------------------------------
    # Music
    # -----------------------------------------------------

    MUSIC_MODEL_NAME: {

        "type":
            "music",

        "category":
            "generation",

        "path":
            MUSIC_MODEL_PATH,

        "repository":
            HF_REPOSITORIES[
                MUSIC_MODEL_NAME
            ],

        "huggingface":
            HF_REPOSITORIES[
                MUSIC_MODEL_NAME
            ],

        "local":
            True,

        "offline":
            True
    },


    # -----------------------------------------------------
    # Video
    # -----------------------------------------------------

    VIDEO_MODEL_NAME: {

        "type":
            "video",

        "category":
            "generation",

        "path":
            VIDEO_MODEL_PATH,

        "repository":
            HF_REPOSITORIES[
                VIDEO_MODEL_NAME
            ],

        "huggingface":
            HF_REPOSITORIES[
                VIDEO_MODEL_NAME
            ],

        "local":
            True,

        "offline":
            True
    },


    # -----------------------------------------------------
    # Image
    # -----------------------------------------------------

    IMAGE_MODEL_NAME: {

        "type":
            "image",

        "category":
            "generation",

        "path":
            IMAGE_MODEL_PATH,

        "repository":
            HF_REPOSITORIES[
                IMAGE_MODEL_NAME
            ],

        "huggingface":
            HF_REPOSITORIES[
                IMAGE_MODEL_NAME
            ],

        "local":
            True,

        "offline":
            True
    },


    # -----------------------------------------------------
    # Image Refiner
    # -----------------------------------------------------

    IMAGE_REFINER_MODEL_NAME: {

        "type":
            "image_refiner",

        "category":
            "generation",

        "path":
            IMAGE_REFINER_MODEL_PATH,

        "repository":
            HF_REPOSITORIES[
                IMAGE_REFINER_MODEL_NAME
            ],

        "huggingface":
            HF_REPOSITORIES[
                IMAGE_REFINER_MODEL_NAME
            ],

        "local":
            True,

        "offline":
            True
    },


    # -----------------------------------------------------
    # Embedding
    # -----------------------------------------------------

    EMBEDDING_MODEL_NAME: {

        "type":
            "embedding",

        "category":
            "memory",

        "path":
            EMBEDDING_MODEL_PATH,

        "repository":
            HF_REPOSITORIES[
                EMBEDDING_MODEL_NAME
            ],

        "huggingface":
            HF_REPOSITORIES[
                EMBEDDING_MODEL_NAME
            ],

        "local":
            True,

        "offline":
            True
    }
}


# =========================================================
# Model Routing
# =========================================================

MODEL_ROLES: Dict[
    str,
    str
] = {

    "general":
        GENERAL_MODEL_NAME,

    "chat":
        GENERAL_MODEL_NAME,

    "text":
        GENERAL_MODEL_NAME,

    "coding":
        CODER_MODEL_NAME,

    "coder":
        CODER_MODEL_NAME,

    "code":
        CODER_MODEL_NAME,

    "vision":
        VISION_MODEL_NAME,

    "visual":
        VISION_MODEL_NAME,

    "reasoning":
        REASONING_MODEL_NAME,

    "think":
        REASONING_MODEL_NAME,

    "speech_to_text":
        STT_MODEL_NAME,

    "stt":
        STT_MODEL_NAME,

    "speech":
        STT_MODEL_NAME,

    "text_to_speech":
        TTS_MODEL_NAME,

    "tts":
        TTS_MODEL_NAME,

    "voice":
        TTS_MODEL_NAME,

    "music":
        MUSIC_MODEL_NAME,

    "video":
        VIDEO_MODEL_NAME,

    "image":
        IMAGE_MODEL_NAME,

    "image_generation":
        IMAGE_MODEL_NAME,

    "image_refiner":
        IMAGE_REFINER_MODEL_NAME,

    "refiner":
        IMAGE_REFINER_MODEL_NAME,

    "embedding":
        EMBEDDING_MODEL_NAME,

    "memory":
        EMBEDDING_MODEL_NAME
}


# =========================================================
# Model Lookup Helpers
# =========================================================

def get_model_name(role: str) -> str:

    if not role:
        return DEFAULT_MODEL

    normalized_role = (
        str(role)
        .strip()
        .lower()
    )

    return MODEL_ROLES.get(
        normalized_role,
        DEFAULT_MODEL
    )


def get_model_config(
    model_name: str
) -> Dict[str, Any]:

    if model_name in LOCAL_MODELS:

        return LOCAL_MODELS[
            model_name
        ]

    role_model = MODEL_ROLES.get(
        str(model_name)
        .strip()
        .lower()
    )

    if role_model in LOCAL_MODELS:

        return LOCAL_MODELS[
            role_model
        ]

    return LOCAL_MODELS[
        DEFAULT_MODEL
    ]


def get_model_repository(
    model_name: str
) -> str:

    config = get_model_config(
        model_name
    )

    return str(
        config["repository"]
    )


def get_model_path(
    model_name: str
) -> str:

    config = get_model_config(
        model_name
    )

    return str(
        config["path"]
    )


def get_all_models() -> Dict[
    str,
    Dict[str, Any]
]:

    return {
        name: config.copy()
        for name, config
        in LOCAL_MODELS.items()
    }


# =========================================================
# Public Configuration
# =========================================================

def get_config() -> Dict[str, Any]:

    return {

        "app": {

            "name":
                "IraAI",

            "offline":
                OFFLINE_MODE,

            "external_ai_api":
                EXTERNAL_AI_API_ENABLED
        },


        "huggingface": {

            "username":
                HF_USERNAME,

            "token_configured":
                bool(HF_TOKEN),

            "repositories":
                HF_REPOSITORIES,

            "download_enabled":
                HF_DOWNLOAD_ENABLED,

            "cache_dir":
                HF_LOCAL_CACHE_DIR,

            "revision":
                HF_REVISION
        },


        "models": {

            "root":
                MODEL_ROOT,

            "default":
                DEFAULT_MODEL,

            "device":
                DEVICE,

            "gpu":
                USE_GPU,

            "local_files_only":
                LOCAL_FILES_ONLY,

            "trust_remote_code":
                TRUST_REMOTE_CODE,

            "available":
                LOCAL_MODELS,

            "roles":
                MODEL_ROLES
        },


        "ai": {

            "temperature":
                AI_TEMPERATURE,

            "max_new_tokens":
                AI_MAX_NEW_TOKENS,

            "top_p":
                AI_TOP_P,

            "do_sample":
                AI_DO_SAMPLE
        },


        "memory": {

            "file":
                MEMORY_FILE,

            "entry_limit":
                MEMORY_ENTRY_LIMIT,

            "auto_delete":
                AUTO_DELETE_MEMORY,

            "search_limit":
                MEMORY_SEARCH_LIMIT
        },


        "tools": {

            "web":
                WEB_TOOLS_ENABLED,

            "files":
                FILE_TOOLS_ENABLED,

            "android":
                ANDROID_TOOLS_ENABLED,

            "images":
                IMAGE_TOOLS_ENABLED,

            "videos":
                VIDEO_TOOLS_ENABLED
        },


        "server": {

            "host":
                HOST,

            "port":
                PORT,

            "debug":
                DEBUG,

            "max_content_length":
                MAX_CONTENT_LENGTH
        },


        "security": {

            "api_key_required":
                API_KEY_REQUIRED
        }
    }


# =========================================================
# Configuration Validation
# =========================================================

def validate_config() -> List[str]:

    errors: List[str] = []


    # -----------------------------------------------------
    # Base Configuration
    # -----------------------------------------------------

    if not BASE_DIR:

        errors.append(
            "Base directory is not configured."
        )


    if not MODEL_ROOT:

        errors.append(
            "Model root directory is empty."
        )


    if not MEMORY_FILE:

        errors.append(
            "Memory file path is empty."
        )


    # -----------------------------------------------------
    # Hugging Face
    # -----------------------------------------------------

    if not HF_USERNAME:

        errors.append(
            "Hugging Face username is not configured."
        )


    if len(HF_REPOSITORIES) != 11:

        errors.append(
            "Exactly 11 Hugging Face repositories "
            "must be configured."
        )


    # -----------------------------------------------------
    # Default Model
    # -----------------------------------------------------

    if not DEFAULT_MODEL:

        errors.append(
            "Default model is not configured."
        )


    if DEFAULT_MODEL not in LOCAL_MODELS:

        errors.append(
            "Default model is not registered."
        )


    # -----------------------------------------------------
    # Model Registry
    # -----------------------------------------------------

    if len(LOCAL_MODELS) != 11:

        errors.append(
            "IraAI model registry must contain "
            "exactly 11 models."
        )


    # -----------------------------------------------------
    # Repository Validation
    # -----------------------------------------------------

    for model_name in LOCAL_MODELS:

        repository = HF_REPOSITORIES.get(
            model_name
        )

        if not repository:

            errors.append(
                "Hugging Face repository is missing "
                f"for model: {model_name}"
            )

            continue


        if "/" not in repository:

            errors.append(
                "Invalid Hugging Face repository "
                f"for model: {model_name}"
            )


    # -----------------------------------------------------
    # Model Path Validation
    # -----------------------------------------------------

    for model_name, config in LOCAL_MODELS.items():

        if not config.get("path"):

            errors.append(
                f"Local path is missing "
                f"for model: {model_name}"
            )


    # -----------------------------------------------------
    # AI Parameters
    # -----------------------------------------------------

    if AI_TEMPERATURE < 0:

        errors.append(
            "AI temperature cannot be negative."
        )


    if AI_MAX_NEW_TOKENS <= 0:

        errors.append(
            "AI max_new_tokens must be "
            "greater than zero."
        )


    if not 0 < AI_TOP_P <= 1:

        errors.append(
            "AI top_p must be between 0 and 1."
        )


    # -----------------------------------------------------
    # CUDA
    # -----------------------------------------------------

    if (
        USE_CUDA == "true"
        and not _cuda_available()
    ):

        errors.append(
            "CUDA was explicitly enabled, "
            "but CUDA is not available."
        )


    # -----------------------------------------------------
    # Server
    # -----------------------------------------------------

    if not 1 <= PORT <= 65535:

        errors.append(
            "Server port must be between "
            "1 and 65535."
        )


    # -----------------------------------------------------
    # Request Size
    # -----------------------------------------------------

    if MAX_CONTENT_LENGTH <= 0:

        errors.append(
            "Maximum content length must "
            "be greater than zero."
        )


    # -----------------------------------------------------
    # Backend API Security
    # -----------------------------------------------------

    if (
        API_KEY_REQUIRED
        and not IRAAI_API_KEY
    ):

        errors.append(
            "Backend API key is required "
            "but not configured."
        )


    return errors


# =========================================================
# END
# =========================================================