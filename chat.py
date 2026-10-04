# =========================================================
# IraAI — Chat Controller
# =========================================================
#
# Render is only the backend/controller layer.
# AI inference is performed by the Remote Inference Server.
#
# Final user-facing answer construction belongs ONLY to:
#     answer_builder.py
#
# No external AI provider is used.
# =========================================================

from __future__ import annotations

import json
import uuid

from typing import Any, Dict, List, Optional

from answer_builder import (
    build_answer,
    build_error_answer,
    get_model_instructions,
)

from models import (
    generate_model_response,
    run_model,
    model_manager,
)

from memory import get_memory_manager
from tools import get_tools_manager

from config import (
    DEFAULT_MODEL,
    MODEL_ROLES,
    REMOTE_MODELS,
)


# =========================================================
# MODEL ROUTE ALIASES
# =========================================================

MODEL_ROUTE_KEYS = {
    "general": "general",
    "chat": "general",
    "text": "general",

    "coder": "coder",
    "coding": "coder",
    "code": "coder",

    "vision": "vision",
    "visual": "vision",
    "image_analysis": "vision",
    "image_analysis_model": "vision",

    "reasoning": "reasoning",
    "advanced_reasoning": "reasoning",
    "think": "reasoning",

    "speech_to_text": "speech_to_text",
    "speech-to-text": "speech_to_text",
    "stt": "speech_to_text",
    "speech": "speech_to_text",

    "text_to_speech": "text_to_speech",
    "text-to-speech": "text_to_speech",
    "tts": "text_to_speech",
    "voice": "text_to_speech",

    "music": "music",
    "music_generation": "music",

    "video": "video",
    "video_generation": "video",

    "image": "image",
    "image_generation": "image",
    "text_to_image": "image",

    "image_refiner": "image_refiner",
    "image_refinement": "image_refiner",
    "refiner": "image_refiner",

    "embedding": "embedding",
    "embeddings": "embedding",
}


# =========================================================
# TEXT MODEL ROLES
# =========================================================

TEXT_ROLES = {
    "general",
    "coder",
    "reasoning",
}


# =========================================================
# CHAT CONTROLLER
# =========================================================

class ChatController:

    def __init__(
        self,
        assistant_name: str = "IraAI",
        memory_manager: Optional[Any] = None,
        tool_manager: Optional[Any] = None,
    ):
        self.assistant_name = assistant_name or "IraAI"

        self.memory_manager = (
            memory_manager or get_memory_manager()
        )

        self.tool_manager = (
            tool_manager or get_tools_manager()
        )

        self.conversations: Dict[
            str,
            List[Dict[str, Any]]
        ] = {}

    # =====================================================
    # SEND MESSAGE
    # =====================================================

    def send_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        model: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Dict[str, Any]:

        message = self._clean_message(message)

        if not message:
            return build_error_answer(
                "Message cannot be empty."
            )

        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        conversation_id = str(
            conversation_id
        ).strip()

        normalized_user_id = (
            str(user_id).strip()
            if user_id is not None
            else None
        )

        self._ensure_conversation_loaded(
            conversation_id=conversation_id,
            user_id=normalized_user_id,
        )

        selected_model, selected_role = (
            self._resolve_model_route(
                message=message,
                model=model,
                role=role,
            )
        )

        memory_context = self._get_memory_context(
            message=message,
            conversation_id=conversation_id,
            user_id=normalized_user_id,
        )

        conversation_context = (
            self._build_conversation_context(
                conversation_id
            )
        )

        tool_context = self._get_tool_context(
            message
        )

        context = self._combine_context(
            conversation_context=conversation_context,
            memory_context=memory_context,
            tool_context=tool_context,
            user_id=normalized_user_id,
        )

        self._add_message(
            conversation_id=conversation_id,
            role="user",
            content=message,
        )

        result = self._execute_model(
            message=message,
            context=context,
            model=selected_model,
            role=selected_role,
        )

        # -------------------------------------------------
        # MODEL ERROR
        # -------------------------------------------------

        if not result.success:

            self._save_to_memory(
                conversation_id=conversation_id,
                role="user",
                content=message,
                user_id=normalized_user_id,
            )

            return build_error_answer(
                result.error
                or "Inference server execution failed.",
                code="INFERENCE_SERVER_ERROR",
            )

        # -------------------------------------------------
        # EMPTY MODEL RESPONSE
        # -------------------------------------------------

        model_response = str(
            result.text or ""
        ).strip()

        if not model_response:

            return build_error_answer(
                "The inference server returned an empty response.",
                code="EMPTY_MODEL_RESPONSE",
            )

        # -------------------------------------------------
        # SAVE ASSISTANT RESPONSE
        # -------------------------------------------------

        self._add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=model_response,
        )

        self._save_to_memory(
            conversation_id=conversation_id,
            role="user",
            content=message,
            user_id=normalized_user_id,
        )

        self._save_to_memory(
            conversation_id=conversation_id,
            role="assistant",
            content=model_response,
            user_id=normalized_user_id,
        )

        # -------------------------------------------------
        # FINAL RESPONSE
        # -------------------------------------------------

        return build_answer(
            model_response=model_response,
            context={
                "conversation_id": conversation_id,
                "user_id": normalized_user_id,
                "model": (
                    result.model
                    or selected_model
                ),
                "role": selected_role,
                "remote": True,
                "remote_inference": True,
                "inference_server": True,
                "local_models": False,
                "external_ai_api": False,
                "tool_context": tool_context,
            },
        )

    # =====================================================
    # RESOLVE MODEL ROUTE
    # =====================================================

    def _resolve_model_route(
        self,
        message: str,
        model: Optional[str],
        role: Optional[str],
    ) -> tuple[str, str]:

        if model:

            model_name = str(
                model
            ).strip()

            if model_name in REMOTE_MODELS:

                detected_role = (
                    self._role_from_model(
                        model_name
                    )
                )

                return (
                    model_name,
                    detected_role or "general",
                )

            normalized_model = (
                model_name
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )

            routed_role = MODEL_ROUTE_KEYS.get(
                normalized_model
            )

            if routed_role:

                routed_model = MODEL_ROLES.get(
                    routed_role
                )

                if routed_model:

                    return (
                        routed_model,
                        routed_role,
                    )

            default_role = (
                self._role_from_model(
                    DEFAULT_MODEL
                )
                or "general"
            )

            return (
                DEFAULT_MODEL,
                default_role,
            )

        if role:

            normalized_role = (
                self._normalize_role(role)
            )

            routed_model = MODEL_ROLES.get(
                normalized_role
            )

            if routed_model:

                return (
                    routed_model,
                    normalized_role,
                )

        detected_role = self._detect_role(
            message
        )

        routed_model = MODEL_ROLES.get(
            detected_role
        )

        if routed_model:

            return (
                routed_model,
                detected_role,
            )

        default_role = (
            self._role_from_model(
                DEFAULT_MODEL
            )
            or "general"
        )

        return (
            DEFAULT_MODEL,
            default_role,
        )

    # =====================================================
    # NORMALIZE ROLE
    # =====================================================

    def _normalize_role(
        self,
        role: str,
    ) -> str:

        normalized = (
            str(role)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        return MODEL_ROUTE_KEYS.get(
            normalized,
            normalized,
        )

    # =====================================================
    # ROLE FROM MODEL
    # =====================================================

    def _role_from_model(
        self,
        model_name: str,
    ) -> Optional[str]:

        for role, configured_model in (
            MODEL_ROLES.items()
        ):
            if configured_model == model_name:
                return role

        return None

    # =====================================================
    # AUTOMATIC ROLE DETECTION
    # =====================================================

    def _detect_role(
        self,
        message: str,
    ) -> str:

        text = (
            str(message)
            .strip()
            .lower()
        )

        coding_terms = (
            "code",
            "coding",
            "program",
            "programming",
            "python",
            "java",
            "kotlin",
            "javascript",
            "typescript",
            "html",
            "css",
            "sql",
            "debug",
            "debugging",
            "bug",
            "error in code",
            "function",
            "class",
            "api",
            "github",
            "gradle",
            "android studio",
            "android app",
            "source code",
        )

        if any(
            term in text
            for term in coding_terms
        ):
            return "coder"

        vision_terms = (
            "analyze image",
            "analyse image",
            "analyze this image",
            "analyse this image",
            "look at this image",
            "look at this photo",
            "what is in this image",
            "what is in this photo",
            "image analysis",
            "photo analysis",
            "picture analysis",
            "describe this image",
            "describe this photo",
        )

        if any(
            term in text
            for term in vision_terms
        ):
            return "vision"

        stt_terms = (
            "speech to text",
            "speech-to-text",
            "transcribe audio",
            "transcribe this audio",
            "transcribe this",
            "transcription",
            "audio transcription",
        )

        if any(
            term in text
            for term in stt_terms
        ):
            return "speech_to_text"

        tts_terms = (
            "text to speech",
            "text-to-speech",
            "read this aloud",
            "speak this",
            "voice this text",
            "convert this text to speech",
        )

        if any(
            term in text
            for term in tts_terms
        ):
            return "text_to_speech"

        music_terms = (
            "generate music",
            "make music",
            "create music",
            "music generation",
            "generate a song",
            "make a song",
            "create a song",
            "musicgen",
        )

        if any(
            term in text
            for term in music_terms
        ):
            return "music"

        video_terms = (
            "generate video",
            "make a video",
            "create a video",
            "video generation",
            "text to video",
            "text-to-video",
            "wan video",
        )

        if any(
            term in text
            for term in video_terms
        ):
            return "video"

        image_refiner_terms = (
            "refine image",
            "refine this image",
            "enhance image",
            "enhance this image",
            "improve image quality",
            "restore image",
            "upscale image",
            "super resolution",
            "image restoration",
        )

        if any(
            term in text
            for term in image_refiner_terms
        ):
            return "image_refiner"

        image_generation_terms = (
            "generate image",
            "create image",
            "make an image",
            "make image",
            "image generation",
            "text to image",
            "text-to-image",
            "draw an image",
            "create a picture",
            "generate a picture",
            "create artwork",
            "generate artwork",
            "sdxl",
        )

        if any(
            term in text
            for term in image_generation_terms
        ):
            return "image"

        embedding_terms = (
            "embedding",
            "embeddings",
            "vector embedding",
            "semantic embedding",
            "create embedding",
            "generate embedding",
        )

        if any(
            term in text
            for term in embedding_terms
        ):
            return "embedding"

        reasoning_terms = (
            "reason",
            "reasoning",
            "think deeply",
            "analyze deeply",
            "analyse deeply",
            "solve this",
            "prove this",
            "derive",
            "step by step",
            "complex problem",
            "mathematical proof",
            "deep reasoning",
        )

        if any(
            term in text
            for term in reasoning_terms
        ):
            return "reasoning"

        return "general"

    # =====================================================
    # EXECUTE REMOTE MODEL
    # =====================================================

    def _execute_model(
        self,
        message: str,
        context: str,
        model: str,
        role: str,
    ):

        # -------------------------------------------------
        # answer_builder.py is the ONLY instruction source.
        # -------------------------------------------------

        instructions = ""

        try:
            instructions = get_model_instructions()
        except Exception:
            instructions = ""

        combined_context = context

        if instructions:

            if combined_context:
                combined_context = (
                    "MODEL INSTRUCTIONS:\n"
                    + str(instructions)
                    + "\n\n"
                    + combined_context
                )
            else:
                combined_context = (
                    "MODEL INSTRUCTIONS:\n"
                    + str(instructions)
                )

        # -------------------------------------------------
        # TEXT MODELS
        # -------------------------------------------------

        if role in TEXT_ROLES:

            return generate_model_response(
                prompt=message,
                model=model,
                context=combined_context,
            )

        # -------------------------------------------------
        # SPECIALIZED REMOTE MODELS
        # -------------------------------------------------

        return run_model(
            model=model,
            prompt=message,
            context=combined_context,
        )

    # =====================================================
    # LOAD CONVERSATION
    # =====================================================

    def _ensure_conversation_loaded(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
    ) -> None:

        if conversation_id in self.conversations:
            return

        try:

            messages = (
                self.memory_manager
                .get_conversation_messages(
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
            )

            if isinstance(messages, list):

                self.conversations[
                    conversation_id
                ] = [

                    {
                        "role": item.get(
                            "role",
                            "unknown",
                        ),
                        "content": item.get(
                            "content",
                            "",
                        ),
                    }

                    for item in messages

                    if isinstance(item, dict)
                ]

                return

        except Exception:
            pass

        self.conversations[
            conversation_id
        ] = []

    # =====================================================
    # ADD MESSAGE
    # =====================================================

    def _add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:

        self.conversations.setdefault(
            conversation_id,
            []
        )

        self.conversations[
            conversation_id
        ].append({
            "role": role,
            "content": content,
        })

    # =====================================================
    # MEMORY CONTEXT
    # =====================================================

    def _get_memory_context(
        self,
        message: str,
        conversation_id: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        try:

            return self.memory_manager.build_context(
                query=message,
                conversation_id=conversation_id,
                user_id=user_id,
            )

        except Exception as error:

            return {
                "memory_error":
                    f"{error.__class__.__name__}: {error}"
            }

    # =====================================================
    # CONVERSATION CONTEXT
    # =====================================================

    def _build_conversation_context(
        self,
        conversation_id: str,
    ) -> str:

        messages = self.conversations.get(
            conversation_id,
            []
        )

        if not messages:
            return ""

        lines: List[str] = []

        for item in messages:

            if not isinstance(item, dict):
                continue

            role = str(
                item.get(
                    "role",
                    "unknown",
                )
            )

            content = str(
                item.get(
                    "content",
                    "",
                )
            )

            if not content:
                continue

            lines.append(
                f"{role.upper()}: {content}"
            )

        return "\n".join(lines)

    # =====================================================
    # TOOL CONTEXT
    # =====================================================

    def _get_tool_context(
        self,
        message: str,
    ) -> Dict[str, Any]:

        if self.tool_manager is None:
            return {}

        try:

            result = self.tool_manager.process(
                request=message
            )

            if isinstance(result, dict):

                if result.get("success"):

                    return self.tool_manager.build_context(
                        [result]
                    )

                return {
                    "tool_result": result
                }

        except Exception as error:

            return {
                "tool_error":
                    f"{error.__class__.__name__}: {error}"
            }

        return {}

    # =====================================================
    # COMBINE CONTEXT
    # =====================================================

    def _combine_context(
        self,
        conversation_context: str,
        memory_context: Dict[str, Any],
        tool_context: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> str:

        sections: List[str] = []

        if user_id:

            sections.append(
                "USER ID:\n"
                + str(user_id)
            )

        if conversation_context:

            sections.append(
                "CONVERSATION HISTORY:\n"
                + conversation_context
            )

        if memory_context:

            try:

                memory_text = json.dumps(
                    memory_context,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

            except Exception:

                memory_text = str(
                    memory_context
                )

            sections.append(
                "MEMORY CONTEXT:\n"
                + memory_text
            )

        if tool_context:

            try:

                tool_text = json.dumps(
                    tool_context,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

            except Exception:

                tool_text = str(
                    tool_context
                )

            sections.append(
                "TOOL CONTEXT:\n"
                + tool_text
            )

        return "\n\n".join(sections)

    # =====================================================
    # SAVE TO MEMORY
    # =====================================================

    def _save_to_memory(
        self,
        conversation_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
    ) -> None:

        try:

            self.memory_manager.save_conversation(
                conversation_id=conversation_id,
                role=role,
                content=content,
                user_id=user_id,
            )

        except Exception:
            pass

    # =====================================================
    # CLEAN MESSAGE
    # =====================================================

    def _clean_message(
        self,
        message: Any,
    ) -> str:

        if message is None:
            return ""

        return str(message).strip()

    # =====================================================
    # GET CONVERSATION
    # =====================================================

    def get_conversation(
        self,
        conversation_id: str,
    ) -> Optional[
        List[Dict[str, Any]]
    ]:

        return self.conversations.get(
            conversation_id
        )

    # =====================================================
    # SET CONVERSATION
    # =====================================================

    def set_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
    ) -> None:

        if not conversation_id:
            return

        if not isinstance(
            messages,
            list
        ):
            messages = []

        self.conversations[
            conversation_id
        ] = messages

    # =====================================================
    # CLEAR CONVERSATION
    # =====================================================

    def clear_conversation(
        self,
        conversation_id: str,
    ) -> bool:

        if conversation_id not in self.conversations:
            return False

        del self.conversations[
            conversation_id
        ]

        return True

    # =====================================================
    # ALL CONVERSATIONS
    # =====================================================

    def get_all_conversations(
        self,
    ) -> Dict[
        str,
        List[Dict[str, Any]]
    ]:

        return self.conversations


# =========================================================
# GLOBAL CONTROLLER
# =========================================================

_chat_controller: Optional[
    ChatController
] = None


def get_chat_controller() -> ChatController:

    global _chat_controller

    if _chat_controller is None:
        _chat_controller = ChatController()

    return _chat_controller


# =========================================================
# PUBLIC CHAT FUNCTION
# =========================================================

def chat(
    message: str,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    model: Optional[str] = None,
    role: Optional[str] = None,
) -> Dict[str, Any]:

    return get_chat_controller().send_message(
        message=message,
        conversation_id=conversation_id,
        user_id=user_id,
        model=model,
        role=role,
    )


# =========================================================
# PUBLIC ROLE CHAT
# =========================================================

def role_chat(
    message: str,
    role: str,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:

    return get_chat_controller().send_message(
        message=message,
        conversation_id=conversation_id,
        user_id=user_id,
        role=role,
    )


# =========================================================
# PUBLIC IMAGE ANALYSIS
# =========================================================

def image_chat(
    image: Any,
    prompt: str = "",
) -> Any:

    try:

        result = model_manager.vision(
            image=image,
            prompt=prompt,
        )

        if not result.success:

            return build_error_answer(
                result.error
                or "Image analysis failed.",
                code="VISION_MODEL_ERROR",
            )

        text = str(
            result.text or ""
        ).strip()

        if not text:

            return build_error_answer(
                "The vision model returned an empty response.",
                code="EMPTY_VISION_RESPONSE",
            )

        return build_answer(
            model_response=text,
            context={
                "model":
                    result.model
                    or "vision",

                "role":
                    "vision",

                "remote":
                    True,

                "remote_inference":
                    True,

                "inference_server":
                    True,

                "local_models":
                    False,

                "external_ai_api":
                    False,
            },
        )

    except Exception as error:

        return build_error_answer(
            f"Image analysis failed: {error}",
            code="VISION_REQUEST_ERROR",
        )


# =========================================================
# END
# =========================================================