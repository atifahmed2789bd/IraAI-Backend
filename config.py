# =========================================================
# IraAI — Remote Inference Configuration
# =========================================================
#
# Render Backend:
#   - API
#   - Chat
#   - Memory
#   - Tools
#   - Model Router
#   - Inference Server communication
#
# Render does NOT:
#   - Download models
#   - Store model weights
#   - Load models
#   - Run model inference
#   - Authenticate with Hugging Face for model loading
#
# Inference Server:
#   - Downloads models from Hugging Face
#   - Authenticates with Hugging Face
#   - Loads models
#   - Runs inference
#
# Final answer construction:
#   - answer_builder.py ONLY
#
# External AI APIs:
#   - NONE
# =========================================================

from __future__ import annotations

import os

from typing import Any, Dict, List


# =========================================================
# Base Paths
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MEMORY_FILE = os.getenv(
    "IRAAI_MEMORY_FILE",
    os.path.join(
        BASE_DIR,
        "iraai_memory.json",
    ),
)


# =========================================================
# Application
# =========================================================

APP_NAME = "IraAI"

EXTERNAL_AI_API_ENABLED = False

# Render communicates with the remote Inference Server.
# Therefore this is NOT an offline network mode.
OFFLINE_MODE = False


# =========================================================
# Inference Server
# =========================================================

INFERENCE_SERVER_URL = os.getenv(
    "INFERENCE_SERVER_URL",
    "",
).strip().rstrip("/")

INFERENCE_SERVER_TOKEN = os.getenv(
    "INFERENCE_SERVER_TOKEN",
    "",
).strip()

INFERENCE_TIMEOUT = float(
    os.getenv(
        "INFERENCE_TIMEOUT",
        "300",
    )
)

INFERENCE_CONNECT_TIMEOUT = float(
    os.getenv(
        "INFERENCE_CONNECT_TIMEOUT",
        "30",
    )
)


# =========================================================
# Inference Endpoints
# =========================================================

INFERENCE_CHAT_ENDPOINT = os.getenv(
    "INFERENCE_CHAT_ENDPOINT",
    "/v1/chat",
).strip()

INFERENCE_VISION_ENDPOINT = os.getenv(
    "INFERENCE_VISION_ENDPOINT",
    "/v1/vision",
).strip()

INFERENCE_STT_ENDPOINT = os.getenv(
    "INFERENCE_STT_ENDPOINT",
    "/v1/speech-to-text",
).strip()

INFERENCE_TTS_ENDPOINT = os.getenv(
    "INFERENCE_TTS_ENDPOINT",
    "/v1/text-to-speech",
).strip()

INFERENCE_MUSIC_ENDPOINT = os.getenv(
    "INFERENCE_MUSIC_ENDPOINT",
    "/v1/music",
).strip()

INFERENCE_VIDEO_ENDPOINT = os.getenv(
    "INFERENCE_VIDEO_ENDPOINT",
    "/v1/video",
).strip()

INFERENCE_IMAGE_ENDPOINT = os.getenv(
    "INFERENCE_IMAGE_ENDPOINT",
    "/v1/image",
).strip()

INFERENCE_IMAGE_REFINER_ENDPOINT = os.getenv(
    "INFERENCE_IMAGE_REFINER_ENDPOINT",
    "/v1/image-refine",
).strip()

INFERENCE_EMBEDDING_ENDPOINT = os.getenv(
    "INFERENCE_EMBEDDING_ENDPOINT",
    "/v1/embedding",
).strip()


# =========================================================
# Hugging Face Account
# =========================================================
#
# Hugging Face is the model source.
#
# IMPORTANT:
# Render does NOT receive or use the Hugging Face token.
#
# HF authentication belongs ONLY on the Inference Server.
#
# Render stores repository identifiers as metadata only.
# =========================================================

HF_USERNAME = os.getenv(
    "IRAAI_HF_USERNAME",
    "atifahmed2789",
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

VIDEO_MODEL_NAME = "Wan2.1-T2V-1.3B"

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

HF_REPOSITORIES: Dict[str, str] = {

    GENERAL_MODEL_NAME:
        f"{HF_USERNAME}/"
        "IraAI-Qwen3-8-27B",

    CODER_MODEL_NAME:
        f"{HF_USERNAME}/"
        "IraAI-Qwen3-Coder-30B-A3B-Instruct",

    VISION_MODEL_NAME:
        f"{HF_USERNAME}/"
        "IraAI-Qwen3-VL-8B-Instruct",

    REASONING_MODEL_NAME:
        f"{HF_USERNAME}/"
        "IraAI-DeepSeek-R1",

    STT_MODEL_NAME:
        f"{HF_USERNAME}/"
        "IraAI-Whisper-Small",

    TTS_MODEL_NAME:
        f"{HF_USERNAME}/"
        "IraAI-Kokoro-82M",

    MUSIC_MODEL_NAME:
        f"{HF_USERNAME}/"
        "IraAI-MusicGen-Small",

    VIDEO_MODEL_NAME:
        f"{HF_USERNAME}/"
        "IraAI-Wan2.1-T2V-1.3B",

    IMAGE_MODEL_NAME:
        f"{HF_USERNAME}/"
        "IraAI-SDXL-Base-1.0",

    IMAGE_REFINER_MODEL_NAME:
        f"{HF_USERNAME}/"
        "IraAI-SDXL-Refiner-1.0",

    EMBEDDING_MODEL_NAME:
        f"{HF_USERNAME}/"
        "IraAI-BGE-M3",
}


# =========================================================
# Backward Compatibility
# =========================================================

HF_MODELS = HF_REPOSITORIES


# =========================================================
# Model Roles
# =========================================================

MODEL_ROLES: Dict[str, str] = {

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
        EMBEDDING_MODEL_NAME,
}


# =========================================================
# Default Model
# =========================================================

DEFAULT_MODEL = os.getenv(
    "IRAAI_DEFAULT_MODEL",
    GENERAL_MODEL_NAME,
)


# =========================================================
# AI Generation Settings
# =========================================================

AI_TEMPERATURE = float(
    os.getenv(
        "IRAAI_TEMPERATURE",
        "0.7",
    )
)

AI_MAX_NEW_TOKENS = int(
    os.getenv(
        "IRAAI_MAX_NEW_TOKENS",
        "2048",
    )
)

AI_TOP_P = float(
    os.getenv(
        "IRAAI_TOP_P",
        "0.9",
    )
)

AI_DO_SAMPLE = (
    os.getenv(
        "IRAAI_DO_SAMPLE",
        "true",
    )
    .strip()
    .lower()
    == "true"
)


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
    "0.0.0.0",
)

PORT = int(
    os.getenv(
        "PORT",
        "8000",
    )
)

DEBUG = (
    os.getenv(
        "IRAAI_DEBUG",
        "false",
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
        ),
    )
)


# =========================================================
# API Security
# =========================================================

API_KEY_REQUIRED = (
    os.getenv(
        "IRAAI_API_KEY_REQUIRED",
        "false",
    )
    .strip()
    .lower()
    == "true"
)

IRAAI_API_KEY = os.getenv(
    "IRAAI_API_KEY",
    "",
)


# =========================================================
# Model Registry
# =========================================================
#
# Metadata only.
#
# Models are NEVER stored or loaded by Render.
# =========================================================

REMOTE_MODELS: Dict[
    str,
    Dict[str, Any],
] = {

    GENERAL_MODEL_NAME: {
        "type": "general",
        "category": "chat",
        "repository":
            HF_REPOSITORIES[
                GENERAL_MODEL_NAME
            ],
        "huggingface":
            HF_REPOSITORIES[
                GENERAL_MODEL_NAME
            ],
        "local": False,
        "remote": True,
    },

    CODER_MODEL_NAME: {
        "type": "coder",
        "category": "coding",
        "repository":
            HF_REPOSITORIES[
                CODER_MODEL_NAME
            ],
        "huggingface":
            HF_REPOSITORIES[
                CODER_MODEL_NAME
            ],
        "local": False,
        "remote": True,
    },

    VISION_MODEL_NAME: {
        "type": "vision",
        "category": "vision",
        "repository":
            HF_REPOSITORIES[
                VISION_MODEL_NAME
            ],
        "huggingface":
            HF_REPOSITORIES[
                VISION_MODEL_NAME
            ],
        "local": False,
        "remote": True,
    },

    REASONING_MODEL_NAME: {
        "type": "reasoning",
        "category": "reasoning",
        "repository":
            HF_REPOSITORIES[
                REASONING_MODEL_NAME
            ],
        "huggingface":
            HF_REPOSITORIES[
                REASONING_MODEL_NAME
            ],
        "local": False,
        "remote": True,
    },

    STT_MODEL_NAME: {
        "type": "speech_to_text",
        "category": "audio",
        "repository":
            HF_REPOSITORIES[
                STT_MODEL_NAME
            ],
        "huggingface":
            HF_REPOSITORIES[
                STT_MODEL_NAME
            ],
        "local": False,
        "remote": True,
    },

    TTS_MODEL_NAME: {
        "type": "text_to_speech",
        "category": "audio",
        "repository":
            HF_REPOSITORIES[
                TTS_MODEL_NAME
            ],
        "huggingface":
            HF_REPOSITORIES[
                TTS_MODEL_NAME
            ],
        "local": False,
        "remote": True,
    },

    MUSIC_MODEL_NAME: {
        "type": "music",
        "category": "generation",
        "repository":
            HF_REPOSITORIES[
                MUSIC_MODEL_NAME
            ],
        "huggingface":
            HF_REPOSITORIES[
                MUSIC_MODEL_NAME
            ],
        "local": False,
        "remote": True,
    },

    VIDEO_MODEL_NAME: {
        "type": "video",
        "category": "generation",
        "repository":
            HF_REPOSITORIES[
                VIDEO_MODEL_NAME
            ],
        "huggingface":
            HF_REPOSITORIES[
                VIDEO_MODEL_NAME
            ],
        "local": False,
        "remote": True,
    },

    IMAGE_MODEL_NAME: {
        "type": "image",
        "category": "generation",
        "repository":
            HF_REPOSITORIES[
                IMAGE_MODEL_NAME
            ],
        "huggingface":
            HF_REPOSITORIES[
                IMAGE_MODEL_NAME
            ],
        "local": False,
        "remote": True,
    },

    IMAGE_REFINER_MODEL_NAME: {
        "type": "image_refiner",
        "category": "generation",
        "repository":
            HF_REPOSITORIES[
                IMAGE_REFINER_MODEL_NAME
            ],
        "huggingface":
            HF_REPOSITORIES[
                IMAGE_REFINER_MODEL_NAME
            ],
        "local": False,
        "remote": True,
    },

    EMBEDDING_MODEL_NAME: {
        "type": "embedding",
        "category": "memory",
        "repository":
            HF_REPOSITORIES[
                EMBEDDING_MODEL_NAME
            ],
        "huggingface":
            HF_REPOSITORIES[
                EMBEDDING_MODEL_NAME
            ],
        "local": False,
        "remote": True,
    },
}


# =========================================================
# Backward Compatibility
# =========================================================

LOCAL_MODELS = REMOTE_MODELS


# =========================================================
# Model Helpers
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
        DEFAULT_MODEL,
    )


def get_model_config(
    model_name: str,
) -> Dict[str, Any]:

    if model_name in REMOTE_MODELS:
        return REMOTE_MODELS[
            model_name
        ]

    role_model = MODEL_ROLES.get(
        str(model_name)
        .strip()
        .lower()
    )

    if role_model in REMOTE_MODELS:
        return REMOTE_MODELS[
            role_model
        ]

    return REMOTE_MODELS[
        DEFAULT_MODEL
    ]


def get_model_repository(
    model_name: str,
) -> str:

    config = get_model_config(
        model_name
    )

    return str(
        config["repository"]
    )


def get_all_models() -> Dict[
    str,
    Dict[str, Any],
]:

    return {
        name: config.copy()
        for name, config
        in REMOTE_MODELS.items()
    }


# =========================================================
# Inference Server Helpers
# =========================================================

def inference_server_configured() -> bool:
    return bool(
        INFERENCE_SERVER_URL
    )


def get_inference_server_url() -> str:
    return INFERENCE_SERVER_URL


def get_inference_server_token() -> str:
    return INFERENCE_SERVER_TOKEN


# =========================================================
# Configuration
# =========================================================

def get_config() -> Dict[str, Any]:

    return {

        "app": {
            "name":
                APP_NAME,

            "offline":
                OFFLINE_MODE,

            "external_ai_api":
                EXTERNAL_AI_API_ENABLED,
        },

        "inference": {

            "server_url":
                INFERENCE_SERVER_URL,

            "configured":
                inference_server_configured(),

            "timeout":
                INFERENCE_TIMEOUT,

            "connect_timeout":
                INFERENCE_CONNECT_TIMEOUT,

            "endpoints": {

                "chat":
                    INFERENCE_CHAT_ENDPOINT,

                "vision":
                    INFERENCE_VISION_ENDPOINT,

                "speech_to_text":
                    INFERENCE_STT_ENDPOINT,

                "text_to_speech":
                    INFERENCE_TTS_ENDPOINT,

                "music":
                    INFERENCE_MUSIC_ENDPOINT,

                "video":
                    INFERENCE_VIDEO_ENDPOINT,

                "image":
                    INFERENCE_IMAGE_ENDPOINT,

                "image_refiner":
                    INFERENCE_IMAGE_REFINER_ENDPOINT,

                "embedding":
                    INFERENCE_EMBEDDING_ENDPOINT,
            },
        },

        "huggingface": {

            "username":
                HF_USERNAME,

            # Token is intentionally NOT handled by Render.
            "token_configured":
                False,

            "authentication_location":
                "inference_server",

            "repositories":
                HF_REPOSITORIES,
        },

        "models": {

            "default":
                DEFAULT_MODEL,

            "available":
                REMOTE_MODELS,

            "roles":
                MODEL_ROLES,

            "local_loading":
                False,

            "remote_inference":
                True,
        },

        "ai": {

            "temperature":
                AI_TEMPERATURE,

            "max_new_tokens":
                AI_MAX_NEW_TOKENS,

            "top_p":
                AI_TOP_P,

            "do_sample":
                AI_DO_SAMPLE,
        },

        "memory": {

            "file":
                MEMORY_FILE,

            "entry_limit":
                MEMORY_ENTRY_LIMIT,

            "auto_delete":
                AUTO_DELETE_MEMORY,

            "search_limit":
                MEMORY_SEARCH_LIMIT,
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
                VIDEO_TOOLS_ENABLED,
        },

        "server": {

            "host":
                HOST,

            "port":
                PORT,

            "debug":
                DEBUG,

            "max_content_length":
                MAX_CONTENT_LENGTH,
        },

        "security": {

            "api_key_required":
                API_KEY_REQUIRED,
        },
    }


# =========================================================
# Configuration Validation
# =========================================================

def validate_config() -> List[str]:

    errors: List[str] = []


    # -----------------------------------------------------
    # Inference Server
    # -----------------------------------------------------

    if not INFERENCE_SERVER_URL:

        errors.append(
            "INFERENCE_SERVER_URL is not configured."
        )


    # -----------------------------------------------------
    # Model Registry
    # -----------------------------------------------------

    if len(REMOTE_MODELS) != 11:

        errors.append(
            "IraAI must have exactly 11 models."
        )


    if len(HF_REPOSITORIES) != 11:

        errors.append(
            "Exactly 11 Hugging Face repositories "
            "must be configured."
        )


    # -----------------------------------------------------
    # Default Model
    # -----------------------------------------------------

    if DEFAULT_MODEL not in REMOTE_MODELS:

        errors.append(
            "Default model is not registered."
        )


    # -----------------------------------------------------
    # Repository Validation
    # -----------------------------------------------------

    for model_name in REMOTE_MODELS:

        repository = HF_REPOSITORIES.get(
            model_name
        )

        if not repository:

            errors.append(
                "Missing Hugging Face repository "
                f"for model: {model_name}"
            )

            continue

        if "/" not in repository:

            errors.append(
                "Invalid Hugging Face repository "
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
    # Inference Timeouts
    # -----------------------------------------------------

    if INFERENCE_TIMEOUT <= 0:

        errors.append(
            "Inference timeout must be "
            "greater than zero."
        )


    if INFERENCE_CONNECT_TIMEOUT <= 0:

        errors.append(
            "Inference connect timeout must be "
            "greater than zero."
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
    # API Security
    # -----------------------------------------------------

    if (
        API_KEY_REQUIRED
        and not IRAAI_API_KEY
    ):

        errors.append(
            "API key is required but not configured."
        )


    return errors


# =========================================================
# END
# =========================================================