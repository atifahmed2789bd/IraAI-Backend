"""
IraAI Remote Model Router

Render responsibilities:
    - Receive model requests
    - Route requests to the inference server
    - Return inference results

NOT responsible for:
    - Loading model weights
    - Running torch/transformers/diffusers
    - Downloading Hugging Face models
    - Building prompts/instructions

Prompt/instruction ownership:
    answer_builder.py

Model execution:
    Separate Inference Server
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =========================================================
# CONFIGURATION
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
    os.getenv("INFERENCE_TIMEOUT", "300")
)

INFERENCE_CONNECT_TIMEOUT = float(
    os.getenv("INFERENCE_CONNECT_TIMEOUT", "30")
)

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
# HUGGING FACE MODEL REPOSITORIES
# =========================================================

HF_REPOSITORIES: Dict[str, str] = {
    "general": "atifahmed2789/IraAI-Qwen3-8-27B",
    "coder": "atifahmed2789/IraAI-Qwen3-Coder-30B-A3B-Instruct",
    "vision": "atifahmed2789/IraAI-Qwen3-VL-8B-Instruct",
    "reasoning": "atifahmed2789/IraAI-DeepSeek-R1",
    "speech_to_text": "atifahmed2789/IraAI-Whisper-Small",
    "text_to_speech": "atifahmed2789/IraAI-Kokoro-82M",
    "music": "atifahmed2789/IraAI-MusicGen-Small",
    "video": "atifahmed2789/IraAI-Wan2.1-T2V-1.3B",
    "image": "atifahmed2789/IraAI-SDXL-Base-1.0",
    "image_refiner": "atifahmed2789/IraAI-SDXL-Refiner-1.0",
    "embedding": "atifahmed2789/IraAI-BGE-M3",
}


# =========================================================
# MODEL NAMES
# =========================================================

MODEL_NAMES: Dict[str, str] = {
    "general": "Qwen3-8-27B",
    "coder": "Qwen3-Coder-30B-A3B-Instruct",
    "vision": "Qwen3-VL-8B-Instruct",
    "reasoning": "DeepSeek-R1",
    "speech_to_text": "Whisper-Small",
    "text_to_speech": "Kokoro-82M",
    "music": "MusicGen-Small",
    "video": "Wan2.1-T2V-1.3B",
    "image": "Stable-Diffusion-XL-Base-1.0",
    "image_refiner": "Stable-Diffusion-XL-Refiner-1.0",
    "embedding": "BGE-M3",
}


# =========================================================
# MODEL RESULT
# =========================================================

@dataclass
class ModelResult:
    success: bool
    text: str = ""
    model: str = ""
    role: str = ""
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "text": self.text,
            "model": self.model,
            "role": self.role,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


# =========================================================
# REMOTE MODEL MANAGER
# =========================================================

class ModelManager:

    def __init__(
        self,
        server_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:

        self.server_url = (
            server_url
            if server_url is not None
            else INFERENCE_SERVER_URL
        ).strip().rstrip("/")

        self.token = (
            token
            if token is not None
            else INFERENCE_SERVER_TOKEN
        ).strip()

        self.timeout = (
            float(timeout)
            if timeout is not None
            else INFERENCE_TIMEOUT
        )

    # =====================================================
    # BASIC
    # =====================================================

    def is_configured(self) -> bool:
        return bool(self.server_url)

    def get_model_repository(
        self,
        role: str,
    ) -> Optional[str]:
        return HF_REPOSITORIES.get(role)

    def get_model_name(
        self,
        role: str,
    ) -> Optional[str]:
        return MODEL_NAMES.get(role)

    def list_models(self) -> Dict[str, Dict[str, str]]:
        return {
            role: {
                "name": MODEL_NAMES[role],
                "repository": HF_REPOSITORIES[role],
            }
            for role in MODEL_NAMES
        }

    def _endpoint(
        self,
        endpoint: str,
    ) -> str:

        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        if not self.server_url:
            return endpoint

        return self.server_url + endpoint

    # =====================================================
    # HTTP REQUEST
    # =====================================================

    def _request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
    ) -> ModelResult:

        if not self.server_url:
            return ModelResult(
                success=False,
                error=(
                    "Inference server is not configured. "
                    "Set INFERENCE_SERVER_URL."
                ),
            )

        url = self._endpoint(endpoint)

        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "IraAI-Render-Backend",
        }

        if self.token:
            headers["Authorization"] = (
                f"Bearer {self.token}"
            )

        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:

                raw = response.read().decode(
                    "utf-8",
                    errors="replace",
                )

                if not raw.strip():
                    return ModelResult(
                        success=False,
                        error="Inference server returned an empty response.",
                    )

                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    result = {
                        "success": True,
                        "text": raw,
                    }

                return self._normalize_result(result)

        except urllib.error.HTTPError as exc:

            try:
                error_body = exc.read().decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                error_body = ""

            return ModelResult(
                success=False,
                error=(
                    f"Inference server HTTP {exc.code}: "
                    f"{error_body or exc.reason}"
                ),
            )

        except urllib.error.URLError as exc:

            return ModelResult(
                success=False,
                error=(
                    "Could not connect to inference server: "
                    f"{exc.reason}"
                ),
            )

        except TimeoutError:

            return ModelResult(
                success=False,
                error="Inference server request timed out.",
            )

        except Exception as exc:

            return ModelResult(
                success=False,
                error=(
                    "Inference server request failed: "
                    f"{exc}"
                ),
            )

    # =====================================================
    # RESULT NORMALIZATION
    # =====================================================

    def _normalize_result(
        self,
        result: Any,
    ) -> ModelResult:

        if isinstance(result, ModelResult):
            return result

        if not isinstance(result, dict):
            return ModelResult(
                success=True,
                data=result,
            )

        success = bool(
            result.get("success", True)
        )

        text = result.get(
            "text",
            result.get(
                "response",
                result.get(
                    "answer",
                    "",
                ),
            ),
        )

        model = str(
            result.get("model", "")
        )

        role = str(
            result.get("role", "")
        )

        error = result.get("error")

        data = result.get(
            "data",
            result.get("result"),
        )

        metadata = result.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            metadata = {
                "value": metadata,
            }

        return ModelResult(
            success=success,
            text=(
                str(text)
                if text is not None
                else ""
            ),
            model=model,
            role=role,
            data=data,
            error=(
                str(error)
                if error is not None
                else None
            ),
            metadata=metadata,
        )

    # =====================================================
    # GENERAL / CODING / REASONING
    # =====================================================

    def generate(
        self,
        prompt: str,
        role: str = "general",
        model: Optional[str] = None,
        conversation: Optional[
            List[Dict[str, Any]]
        ] = None,
        context: Optional[
            Dict[str, Any]
        ] = None,
        **kwargs: Any,
    ) -> ModelResult:

        role = str(role).strip().lower()

        if role not in (
            "general",
            "coder",
            "reasoning",
        ):
            role = "general"

        selected_model = (
            model
            or MODEL_NAMES[role]
        )

        payload: Dict[str, Any] = {
            "role": role,
            "model": selected_model,
            "repository": HF_REPOSITORIES[role],
            "prompt": prompt,
            "conversation": (
                conversation
                if conversation is not None
                else []
            ),
            "context": (
                context
                if context is not None
                else {}
            ),
        }

        payload.update(kwargs)

        return self._request(
            INFERENCE_CHAT_ENDPOINT,
            payload,
        )

    # =====================================================
    # VISION
    # =====================================================

    def vision(
        self,
        prompt: str = "",
        image: Any = None,
        images: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> ModelResult:

        payload: Dict[str, Any] = {
            "role": "vision",
            "model": MODEL_NAMES["vision"],
            "repository": HF_REPOSITORIES["vision"],
            "prompt": prompt,
            "image": image,
            "images": (
                images
                if images is not None
                else []
            ),
        }

        payload.update(kwargs)

        return self._request(
            INFERENCE_VISION_ENDPOINT,
            payload,
        )

    # =====================================================
    # SPEECH TO TEXT
    # =====================================================

    def speech_to_text(
        self,
        audio: Any = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResult:

        payload: Dict[str, Any] = {
            "role": "speech_to_text",
            "model": MODEL_NAMES["speech_to_text"],
            "repository": HF_REPOSITORIES["speech_to_text"],
            "audio": audio,
            "language": language,
        }

        payload.update(kwargs)

        return self._request(
            INFERENCE_STT_ENDPOINT,
            payload,
        )

    # =====================================================
    # TEXT TO SPEECH
    # =====================================================

    def text_to_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResult:

        payload: Dict[str, Any] = {
            "role": "text_to_speech",
            "model": MODEL_NAMES["text_to_speech"],
            "repository": HF_REPOSITORIES["text_to_speech"],
            "text": text,
            "voice": voice,
            "language": language,
        }

        payload.update(kwargs)

        return self._request(
            INFERENCE_TTS_ENDPOINT,
            payload,
        )

    # =====================================================
    # MUSIC
    # =====================================================

    def music(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> ModelResult:

        payload: Dict[str, Any] = {
            "role": "music",
            "model": MODEL_NAMES["music"],
            "repository": HF_REPOSITORIES["music"],
            "prompt": prompt,
        }

        payload.update(kwargs)

        return self._request(
            INFERENCE_MUSIC_ENDPOINT,
            payload,
        )

    # =====================================================
    # VIDEO
    # =====================================================

    def video(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> ModelResult:

        payload: Dict[str, Any] = {
            "role": "video",
            "model": MODEL_NAMES["video"],
            "repository": HF_REPOSITORIES["video"],
            "prompt": prompt,
        }

        payload.update(kwargs)

        return self._request(
            INFERENCE_VIDEO_ENDPOINT,
            payload,
        )

    # =====================================================
    # IMAGE
    # =====================================================

    def image(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> ModelResult:

        payload: Dict[str, Any] = {
            "role": "image",
            "model": MODEL_NAMES["image"],
            "repository": HF_REPOSITORIES["image"],
            "prompt": prompt,
        }

        payload.update(kwargs)

        return self._request(
            INFERENCE_IMAGE_ENDPOINT,
            payload,
        )

    # =====================================================
    # IMAGE REFINEMENT
    # =====================================================

    def refine_image(
        self,
        image: Any = None,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResult:

        payload: Dict[str, Any] = {
            "role": "image_refiner",
            "model": MODEL_NAMES["image_refiner"],
            "repository": HF_REPOSITORIES["image_refiner"],
            "image": image,
            "prompt": prompt,
        }

        payload.update(kwargs)

        return self._request(
            INFERENCE_IMAGE_REFINER_ENDPOINT,
            payload,
        )

    # =====================================================
    # EMBEDDING
    # =====================================================

    def embedding(
        self,
        text: Any,
        **kwargs: Any,
    ) -> ModelResult:

        payload: Dict[str, Any] = {
            "role": "embedding",
            "model": MODEL_NAMES["embedding"],
            "repository": HF_REPOSITORIES["embedding"],
            "text": text,
        }

        payload.update(kwargs)

        return self._request(
            INFERENCE_EMBEDDING_ENDPOINT,
            payload,
        )

    # =====================================================
    # GENERIC ROUTING
    # =====================================================

    def route(
        self,
        role: str,
        **kwargs: Any,
    ) -> ModelResult:

        role = str(role).strip().lower()

        if role in (
            "general",
            "coder",
            "reasoning",
        ):
            return self.generate(
                role=role,
                **kwargs,
            )

        if role == "vision":
            return self.vision(**kwargs)

        if role == "speech_to_text":
            return self.speech_to_text(**kwargs)

        if role == "text_to_speech":
            return self.text_to_speech(**kwargs)

        if role == "music":
            return self.music(**kwargs)

        if role == "video":
            return self.video(**kwargs)

        if role == "image":
            return self.image(**kwargs)

        if role == "image_refiner":
            return self.refine_image(**kwargs)

        if role == "embedding":
            return self.embedding(**kwargs)

        return ModelResult(
            success=False,
            role=role,
            error=f"Unknown model role: {role}",
        )


# =========================================================
# GLOBAL MODEL MANAGER
# =========================================================

model_manager = ModelManager()


# =========================================================
# PUBLIC FUNCTIONS
# =========================================================

def get_model_manager() -> ModelManager:
    return model_manager


def get_available_models() -> Dict[str, Dict[str, str]]:
    return model_manager.list_models()


def generate_ai_response(
    prompt: str,
    role: str = "general",
    **kwargs: Any,
) -> ModelResult:

    return model_manager.generate(
        prompt=prompt,
        role=role,
        **kwargs,
    )


def generate_model_response(
    prompt: str,
    role: str = "general",
    model: Optional[str] = None,
    conversation: Optional[
        List[Dict[str, Any]]
    ] = None,
    context: Optional[
        Dict[str, Any]
    ] = None,
    **kwargs: Any,
) -> ModelResult:
    """
    Compatibility wrapper used by chat.py.

    All actual inference is performed by the
    separate inference server.
    """

    return model_manager.generate(
        prompt=prompt,
        role=role,
        model=model,
        conversation=conversation,
        context=context,
        **kwargs,
    )


def run_model(
    prompt: str = "",
    role: str = "general",
    model: Optional[str] = None,
    conversation: Optional[
        List[Dict[str, Any]]
    ] = None,
    context: Optional[
        Dict[str, Any]
    ] = None,
    **kwargs: Any,
) -> ModelResult:
    """
    Generic compatibility wrapper used by chat.py.
    """

    return model_manager.route(
        role=role,
        prompt=prompt,
        model=model,
        conversation=conversation,
        context=context,
        **kwargs,
    )


def analyze_image(
    prompt: str = "",
    image: Any = None,
    images: Optional[List[Any]] = None,
    **kwargs: Any,
) -> ModelResult:
    """
    Compatibility wrapper for vision requests.
    """

    return model_manager.vision(
        prompt=prompt,
        image=image,
        images=images,
        **kwargs,
    )


# =========================================================
# SPECIALIZED PUBLIC FUNCTIONS
# =========================================================

def speech_to_text(
    audio: Any = None,
    language: Optional[str] = None,
    **kwargs: Any,
) -> ModelResult:

    return model_manager.speech_to_text(
        audio=audio,
        language=language,
        **kwargs,
    )


def text_to_speech(
    text: str,
    voice: Optional[str] = None,
    language: Optional[str] = None,
    **kwargs: Any,
) -> ModelResult:

    return model_manager.text_to_speech(
        text=text,
        voice=voice,
        language=language,
        **kwargs,
    )


def generate_music(
    prompt: str,
    **kwargs: Any,
) -> ModelResult:

    return model_manager.music(
        prompt=prompt,
        **kwargs,
    )


def generate_video(
    prompt: str,
    **kwargs: Any,
) -> ModelResult:

    return model_manager.video(
        prompt=prompt,
        **kwargs,
    )


def generate_image(
    prompt: str,
    **kwargs: Any,
) -> ModelResult:

    return model_manager.image(
        prompt=prompt,
        **kwargs,
    )


def refine_image(
    image: Any = None,
    prompt: Optional[str] = None,
    **kwargs: Any,
) -> ModelResult:

    return model_manager.refine_image(
        image=image,
        prompt=prompt,
        **kwargs,
    )


def create_embedding(
    text: Any,
    **kwargs: Any,
) -> ModelResult:

    return model_manager.embedding(
        text=text,
        **kwargs,
    )


# =========================================================
# HEALTH / STATUS
# =========================================================

def inference_server_configured() -> bool:
    return model_manager.is_configured()


def get_inference_server_url() -> str:
    return model_manager.server_url


def get_model_status() -> Dict[str, Any]:
    return {
        "configured": model_manager.is_configured(),
        "server_url": model_manager.server_url,
        "remote_inference": True,
        "local_model_loading": False,
        "external_ai_api": False,
        "models": model_manager.list_models(),
    }


def model_health() -> Dict[str, Any]:
    """
    Lightweight status check.

    Does not load or execute any model on Render.
    """

    if not model_manager.is_configured():
        return {
            "success": False,
            "status": "not_configured",
            "remote_inference": True,
            "local_model_loading": False,
            "external_ai_api": False,
            "error": (
                "INFERENCE_SERVER_URL is not configured."
            ),
        }

    return {
        "success": True,
        "status": "configured",
        "remote_inference": True,
        "local_model_loading": False,
        "external_ai_api": False,
        "server_url": model_manager.server_url,
        "models": list(MODEL_NAMES.keys()),
    }