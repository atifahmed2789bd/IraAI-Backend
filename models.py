# =========================================================
# IraAI — Local Model Manager
# =========================================================
#
# Model source:
#     Hugging Face Repositories
#
# Inference:
#     LOCAL ONLY
#
# Final answer construction:
#     answer_builder.py ONLY
#
# Supported models:
#     1. Qwen3-8-27B
#     2. Qwen3-Coder-30B-A3B-Instruct
#     3. Qwen3-VL-8B-Instruct
#     4. DeepSeek-R1
#     5. Whisper-Small
#     6. Kokoro-82M
#     7. MusicGen-Small
#     8. Wan2.1-T2V-1.3B
#     9. Stable-Diffusion-XL-Base-1.0
#    10. Stable-Diffusion-XL-Refiner-1.0
#    11. BGE-M3
# =========================================================

from __future__ import annotations

import gc
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from answer_builder import get_model_instructions

from config import (
    AI_DO_SAMPLE,
    AI_MAX_NEW_TOKENS,
    AI_TEMPERATURE,
    AI_TOP_P,
    DEFAULT_MODEL,
    DEVICE,
    HF_DOWNLOAD_ENABLED,
    HF_REPOSITORIES,
    HF_RESUME_DOWNLOAD,
    HF_REVISION,
    HF_TOKEN,
    LOCAL_MODELS,
    MODEL_ROLES,
    TRUST_REMOTE_CODE,
    USE_GPU,
)


# =========================================================
# RESULT
# =========================================================

@dataclass
class ModelResult:
    success: bool
    text: str = ""
    model: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    data: Any = None


# =========================================================
# BASE MODEL ADAPTER
# =========================================================

class ModelAdapter:

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
    ):
        self.name = name
        self.config = config

        self.model_type = str(
            config.get("type", "unknown")
        )

        self.category = str(
            config.get("category", "unknown")
        )

        self.path = str(
            config.get("path", "")
        )

        self.repository = str(
            config.get(
                "repository",
                config.get("huggingface", ""),
            )
        )

        self.huggingface = self.repository

        self.loaded = False
        self.loading = False

        self.lock = threading.RLock()

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.pipeline = None

    # =====================================================
    # LOCAL PATH
    # =====================================================

    def exists(self) -> bool:
        return bool(
            self.path
            and os.path.isdir(self.path)
        )

    # =====================================================
    # SOURCE
    # =====================================================

    def source(self) -> str:
        return self.repository

    # =====================================================
    # ENSURE LOCAL
    # =====================================================

    def ensure_local(self) -> str:

        if self.exists():
            return self.path

        if not HF_DOWNLOAD_ENABLED:
            raise FileNotFoundError(
                "Local model is missing and "
                "Hugging Face download is disabled: "
                f"{self.path}"
            )

        if not self.repository:
            raise RuntimeError(
                f"No Hugging Face repository configured "
                f"for model: {self.name}"
            )

        try:
            from huggingface_hub import snapshot_download
        except Exception as error:
            raise RuntimeError(
                "huggingface_hub is required: "
                f"{error}"
            ) from error

        os.makedirs(
            self.path,
            exist_ok=True,
        )

        kwargs: Dict[str, Any] = {
            "repo_id": self.repository,
            "revision": HF_REVISION,
            "local_dir": self.path,
        }

        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        if HF_RESUME_DOWNLOAD:
            kwargs["resume_download"] = True

        try:
            snapshot_download(**kwargs)

        except TypeError:
            kwargs.pop(
                "resume_download",
                None,
            )
            snapshot_download(**kwargs)

        if not self.exists():
            raise RuntimeError(
                "Model download completed but "
                f"local directory is missing: {self.path}"
            )

        return self.path

    # =====================================================
    # LOAD
    # =====================================================

    def load(self) -> None:
        raise NotImplementedError

    # =====================================================
    # UNLOAD
    # =====================================================

    def unload(self) -> None:

        with self.lock:

            self.model = None
            self.tokenizer = None
            self.processor = None
            self.pipeline = None

            self.loaded = False
            self.loading = False

            gc.collect()

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception:
                pass

    # =====================================================
    # DEVICE
    # =====================================================

    def runtime_device(self):

        if USE_GPU:
            return DEVICE

        return "cpu"

    # =====================================================
    # HEALTH
    # =====================================================

    def health(self) -> Dict[str, Any]:

        return {
            "name": self.name,
            "type": self.model_type,
            "category": self.category,
            "path": self.path,
            "repository": self.repository,
            "source": self.repository,
            "exists": self.exists(),
            "loaded": self.loaded,
            "local": True,
            "offline": True,
            "external_api": False,
        }


# =========================================================
# TEXT MODEL
# =========================================================

class TextModelAdapter(ModelAdapter):

    def load(self) -> None:

        with self.lock:

            if self.loaded:
                return

            if self.loading:
                raise RuntimeError(
                    "Text model is already loading."
                )

            self.loading = True

            try:

                model_path = self.ensure_local()

                try:
                    import torch

                    from transformers import (
                        AutoModelForCausalLM,
                        AutoTokenizer,
                    )

                except Exception as error:
                    raise RuntimeError(
                        "Text model dependencies are "
                        f"not available: {error}"
                    ) from error

                self.tokenizer = (
                    AutoTokenizer.from_pretrained(
                        model_path,
                        local_files_only=True,
                        trust_remote_code=TRUST_REMOTE_CODE,
                    )
                )

                dtype = (
                    torch.float16
                    if USE_GPU
                    else torch.float32
                )

                load_kwargs: Dict[str, Any] = {
                    "local_files_only": True,
                    "trust_remote_code": TRUST_REMOTE_CODE,
                    "torch_dtype": dtype,
                }

                if USE_GPU:
                    load_kwargs["device_map"] = "auto"

                self.model = (
                    AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **load_kwargs,
                    )
                )

                if not USE_GPU:
                    self.model.to("cpu")

                if (
                    self.tokenizer.pad_token_id
                    is None
                    and self.tokenizer.eos_token_id
                    is not None
                ):
                    self.tokenizer.pad_token = (
                        self.tokenizer.eos_token
                    )

                self.model.eval()
                self.loaded = True

            finally:
                self.loading = False

    # =====================================================
    # DEVICE
    # =====================================================

    def _get_input_device(self):

        if self.model is None:
            return "cpu"

        try:
            return next(
                self.model.parameters()
            ).device

        except StopIteration:
            return (
                "cuda"
                if USE_GPU
                else "cpu"
            )

    # =====================================================
    # PROMPT TOKENIZATION
    # =====================================================

    def _tokenize(self, prompt: str):

        if (
            hasattr(
                self.tokenizer,
                "apply_chat_template",
            )
            and self.tokenizer.chat_template
        ):

            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            try:

                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

            except Exception:
                pass

        return self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

    # =====================================================
    # GENERATE
    # =====================================================

    def generate(
        self,
        prompt: str,
        temperature: float = AI_TEMPERATURE,
        max_new_tokens: int = AI_MAX_NEW_TOKENS,
        top_p: float = AI_TOP_P,
        do_sample: bool = AI_DO_SAMPLE,
    ) -> str:

        self.load()

        if self.model is None:
            raise RuntimeError(
                "Text model is not loaded."
            )

        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer is not loaded."
            )

        prompt = str(prompt).strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        with self.lock:

            inputs = self._tokenize(prompt)

            device = self._get_input_device()

            # -------------------------------------------------
            # Tokenized input handling
            # -------------------------------------------------

            if hasattr(inputs, "to"):

                input_ids = inputs.to(device)

                model_inputs = {
                    "input_ids": input_ids
                }

            else:

                model_inputs = {
                    key: value.to(device)
                    if hasattr(value, "to")
                    else value
                    for key, value in inputs.items()
                }

                input_ids = model_inputs.get(
                    "input_ids"
                )

            if input_ids is None:
                raise RuntimeError(
                    "Tokenizer did not return input_ids."
                )

            # -------------------------------------------------
            # Generation arguments
            # -------------------------------------------------

            generation_kwargs: Dict[str, Any] = {
                "max_new_tokens": max(
                    1,
                    int(max_new_tokens),
                ),
                "repetition_penalty": 1.05,
                "do_sample": bool(do_sample),
            }

            if self.tokenizer.pad_token_id is not None:
                generation_kwargs["pad_token_id"] = (
                    self.tokenizer.pad_token_id
                )

            if self.tokenizer.eos_token_id is not None:
                generation_kwargs["eos_token_id"] = (
                    self.tokenizer.eos_token_id
                )

            # Sampling parameters are kept at the same
            # indentation level to avoid deployment syntax
            # and indentation errors.

            generation_kwargs["temperature"] = max(
                0.01,
                float(temperature),
            )

            generation_kwargs["top_p"] = min(
                1.0,
                max(
                    0.01,
                    float(top_p),
                ),
            )

            # -------------------------------------------------
            # PyTorch generation
            # -------------------------------------------------

            try:
                import torch

            except Exception as error:
                raise RuntimeError(
                    f"PyTorch is required: {error}"
                ) from error

            with torch.inference_mode():

                output = self.model.generate(
                    **model_inputs,
                    **generation_kwargs,
                )

            # -------------------------------------------------
            # Remove input tokens
            # -------------------------------------------------

            input_length = (
                input_ids.shape[-1]
            )

            generated_tokens = (
                output[0][input_length:]
            )

            # -------------------------------------------------
            # Decode
            # -------------------------------------------------

            result = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            return result.strip()


# =========================================================
# VISION MODEL — QWEN3-VL
# =========================================================

class VisionModelAdapter(ModelAdapter):

    def load(self) -> None:

        with self.lock:

            if self.loaded:
                return

            if self.loading:
                raise RuntimeError(
                    "Vision model is already loading."
                )

            self.loading = True

            try:

                model_path = self.ensure_local()

                try:
                    import torch

                    from transformers import (
                        AutoProcessor,
                    )

                    try:
                        from transformers import (
                            Qwen3VLForConditionalGeneration
                        )

                        model_class = (
                            Qwen3VLForConditionalGeneration
                        )

                    except ImportError:

                        from transformers import (
                            AutoModelForImageTextToText
                        )

                        model_class = (
                            AutoModelForImageTextToText
                        )

                except Exception as error:
                    raise RuntimeError(
                        "Vision dependencies are "
                        f"not available: {error}"
                    ) from error

                self.processor = (
                    AutoProcessor.from_pretrained(
                        model_path,
                        local_files_only=True,
                        trust_remote_code=TRUST_REMOTE_CODE,
                    )
                )

                dtype = (
                    torch.float16
                    if USE_GPU
                    else torch.float32
                )

                load_kwargs: Dict[str, Any] = {
                    "local_files_only": True,
                    "trust_remote_code": TRUST_REMOTE_CODE,
                    "torch_dtype": dtype,
                }

                if USE_GPU:
                    load_kwargs["device_map"] = "auto"

                self.model = (
                    model_class.from_pretrained(
                        model_path,
                        **load_kwargs,
                    )
                )

                if not USE_GPU:
                    self.model.to("cpu")

                self.model.eval()
                self.loaded = True

            finally:
                self.loading = False

    # =====================================================
    # IMAGE PREPARATION
    # =====================================================

    @staticmethod
    def _prepare_image(image: Any):

        if isinstance(image, str):

            if not os.path.isfile(image):
                raise FileNotFoundError(
                    f"Image file not found: {image}"
                )

            from PIL import Image

            return Image.open(
                image
            ).convert("RGB")

        return image

    # =====================================================
    # GENERATE
    # =====================================================

    def generate(
        self,
        prompt: str,
        image: Any = None,
        max_new_tokens: int = AI_MAX_NEW_TOKENS,
    ) -> ModelResult:

        if image is None:
            return ModelResult(
                success=False,
                model=self.name,
                error=(
                    "Vision model requires "
                    "an image input."
                ),
            )

        try:

            self.load()

            if self.processor is None:
                raise RuntimeError(
                    "Vision processor is not loaded."
                )

            if self.model is None:
                raise RuntimeError(
                    "Vision model is not loaded."
                )

            image = self._prepare_image(image)

            prompt = str(prompt).strip()

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

            if hasattr(
                self.processor,
                "apply_chat_template",
            ):

                inputs = (
                    self.processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                )

            else:

                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                )

            try:
                device = next(
                    self.model.parameters()
                ).device

            except StopIteration:
                device = (
                    "cuda"
                    if USE_GPU
                    else "cpu"
                )

            inputs = {
                key: value.to(device)
                if hasattr(value, "to")
                else value
                for key, value in inputs.items()
            }

            import torch

            with torch.inference_mode():

                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max(
                        1,
                        int(max_new_tokens),
                    ),
                )

            input_ids = inputs.get(
                "input_ids"
            )

            if input_ids is not None:

                trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids
                    in zip(
                        input_ids,
                        output,
                    )
                ]

                text = self.processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

            else:

                text = self.processor.batch_decode(
                    output,
                    skip_special_tokens=True,
                )[0]

            return ModelResult(
                success=True,
                text=text.strip(),
                model=self.name,
                metadata={
                    "type": "vision",
                    "category": self.category,
                    "repository": self.repository,
                    "local": True,
                    "offline": True,
                    "external_api": False,
                },
            )

        except Exception as error:

            return ModelResult(
                success=False,
                model=self.name,
                error=(
                    f"{error.__class__.__name__}: "
                    f"{error}"
                ),
            )


# =========================================================
# WHISPER — SPEECH TO TEXT
# =========================================================

class WhisperModelAdapter(ModelAdapter):

    def load(self) -> None:

        with self.lock:

            if self.loaded:
                return

            model_path = self.ensure_local()

            try:
                import torch

                from transformers import (
                    AutoProcessor,
                    AutoModelForSpeechSeq2Seq,
                )

            except Exception as error:
                raise RuntimeError(
                    "Whisper dependencies are "
                    f"not available: {error}"
                ) from error

            self.processor = (
                AutoProcessor.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=TRUST_REMOTE_CODE,
                )
            )

            dtype = (
                torch.float16
                if USE_GPU
                else torch.float32
            )

            kwargs: Dict[str, Any] = {
                "local_files_only": True,
                "trust_remote_code": TRUST_REMOTE_CODE,
                "torch_dtype": dtype,
            }

            if USE_GPU:
                kwargs["device_map"] = "auto"

            self.model = (
                AutoModelForSpeechSeq2Seq
                .from_pretrained(
                    model_path,
                    **kwargs,
                )
            )

            if not USE_GPU:
                self.model.to("cpu")

            self.model.eval()
            self.loaded = True

    # =====================================================
    # TRANSCRIBE
    # =====================================================

    def transcribe(
        self,
        audio: Any,
        language: Optional[str] = None,
    ) -> ModelResult:

        try:

            self.load()

            import numpy as np

            if isinstance(audio, str):

                if not os.path.isfile(audio):
                    raise FileNotFoundError(
                        f"Audio file not found: {audio}"
                    )

                try:
                    import soundfile as sf

                    audio_array, sampling_rate = (
                        sf.read(audio)
                    )

                except Exception as error:
                    raise RuntimeError(
                        "soundfile is required for "
                        f"audio files: {error}"
                    ) from error

            elif isinstance(audio, tuple):

                audio_array, sampling_rate = audio

            else:

                audio_array = audio
                sampling_rate = 16000

            audio_array = np.asarray(
                audio_array
            )

            if audio_array.ndim > 1:
                audio_array = audio_array.mean(
                    axis=1
                )

            inputs = self.processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            )

            try:
                device = next(
                    self.model.parameters()
                ).device

            except StopIteration:
                device = (
                    "cuda"
                    if USE_GPU
                    else "cpu"
                )

            inputs = {
                key: value.to(device)
                if hasattr(value, "to")
                else value
                for key, value in inputs.items()
            }

            import torch

            generate_kwargs: Dict[str, Any] = {}

            if language:
                generate_kwargs["language"] = language

            with torch.inference_mode():

                generated = self.model.generate(
                    **inputs,
                    **generate_kwargs,
                )

            text = self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
            )[0]

            return ModelResult(
                success=True,
                text=text.strip(),
                model=self.name,
                data=text.strip(),
                metadata={
                    "type": "speech_to_text",
                    "local": True,
                    "offline": True,
                    "external_api": False,
                },
            )

        except Exception as error:

            return ModelResult(
                success=False,
                model=self.name,
                error=(
                    f"{error.__class__.__name__}: "
                    f"{error}"
                ),
            )

    def run(self, **kwargs) -> ModelResult:

        audio = kwargs.get(
            "audio",
            kwargs.get("input"),
        )

        language = kwargs.get(
            "language"
        )

        if audio is None:
            return ModelResult(
                success=False,
                model=self.name,
                error="Audio input is required.",
            )

        return self.transcribe(
            audio=audio,
            language=language,
        )


# =========================================================
# KOKORO — TEXT TO SPEECH
# =========================================================

class KokoroModelAdapter(ModelAdapter):

    def load(self) -> None:

        with self.lock:

            if self.loaded:
                return

            model_path = self.ensure_local()

            try:
                from kokoro import KPipeline

            except Exception as error:
                raise RuntimeError(
                    "Kokoro runtime is not installed: "
                    f"{error}"
                ) from error

            language = self.config.get(
                "language",
                "a",
            )

            self.pipeline = KPipeline(
                lang_code=language,
                repo_id=model_path,
            )

            self.loaded = True

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> ModelResult:

        try:

            self.load()

            if not text or not str(text).strip():
                raise ValueError(
                    "Text cannot be empty."
                )

            if self.pipeline is None:
                raise RuntimeError(
                    "Kokoro pipeline is not loaded."
                )

            if output_path is None:
                output_path = os.path.join(
                    os.getcwd(),
                    "iraai_tts.wav",
                )

            os.makedirs(
                os.path.dirname(
                    os.path.abspath(output_path)
                ),
                exist_ok=True,
            )

            import soundfile as sf

            selected_voice = (
                voice
                or self.config.get(
                    "voice",
                    "af_heart",
                )
            )

            samples = []

            generator = self.pipeline(
                str(text),
                voice=selected_voice,
                speed=float(speed),
            )

            for _, _, audio in generator:
                samples.append(audio)

            if not samples:
                raise RuntimeError(
                    "Kokoro generated no audio."
                )

            import numpy as np

            audio_data = np.concatenate(
                [
                    np.asarray(sample)
                    for sample in samples
                ]
            )

            sample_rate = int(
                getattr(
                    self.pipeline,
                    "sample_rate",
                    24000,
                )
            )

            sf.write(
                output_path,
                audio_data,
                sample_rate,
            )

            return ModelResult(
                success=True,
                model=self.name,
                data=output_path,
                metadata={
                    "type": "text_to_speech",
                    "path": output_path,
                    "sample_rate": sample_rate,
                    "local": True,
                    "offline": True,
                    "external_api": False,
                },
            )

        except Exception as error:

            return ModelResult(
                success=False,
                model=self.name,
                error=(
                    f"{error.__class__.__name__}: "
                    f"{error}"
                ),
            )

    def run(self, **kwargs) -> ModelResult:

        text = kwargs.get(
            "text",
            kwargs.get("input"),
        )

        if text is None:
            return ModelResult(
                success=False,
                model=self.name,
                error="Text input is required.",
            )

        return self.synthesize(
            text=str(text),
            output_path=kwargs.get(
                "output_path"
            ),
            voice=kwargs.get(
                "voice"
            ),
            speed=float(
                kwargs.get(
                    "speed",
                    1.0,
                )
            ),
        )


# =========================================================
# MUSICGEN
# =========================================================

class MusicGenModelAdapter(ModelAdapter):

    def load(self) -> None:

        with self.lock:

            if self.loaded:
                return

            model_path = self.ensure_local()

            try:
                import torch

                from transformers import (
                    AutoProcessor,
                    MusicgenForConditionalGeneration,
                )

            except Exception as error:
                raise RuntimeError(
                    "MusicGen dependencies are "
                    f"not available: {error}"
                ) from error

            self.processor = (
                AutoProcessor.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=TRUST_REMOTE_CODE,
                )
            )

            dtype = (
                torch.float16
                if USE_GPU
                else torch.float32
            )

            kwargs: Dict[str, Any] = {
                "local_files_only": True,
                "trust_remote_code": TRUST_REMOTE_CODE,
                "torch_dtype": dtype,
            }

            if USE_GPU:
                kwargs["device_map"] = "auto"

            self.model = (
                MusicgenForConditionalGeneration
                .from_pretrained(
                    model_path,
                    **kwargs,
                )
            )

            if not USE_GPU:
                self.model.to("cpu")

            self.model.eval()
            self.loaded = True

    def generate_music(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> ModelResult:

        try:

            self.load()

            if output_path is None:
                output_path = os.path.join(
                    os.getcwd(),
                    "iraai_music.wav",
                )

            inputs = self.processor(
                text=[str(prompt)],
                padding=True,
                return_tensors="pt",
            )

            try:
                device = next(
                    self.model.parameters()
                ).device

            except StopIteration:
                device = (
                    "cuda"
                    if USE_GPU
                    else "cpu"
                )

            inputs = {
                key: value.to(device)
                if hasattr(value, "to")
                else value
                for key, value in inputs.items()
            }

            import torch

            with torch.inference_mode():

                audio_values = self.model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=3.0,
                    max_new_tokens=max(
                        1,
                        int(max_new_tokens),
                    ),
                )

            audio = (
                audio_values[0, 0]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            sample_rate = int(
                self.model.config
                .audio_encoder
                .sampling_rate
            )

            import soundfile as sf

            os.makedirs(
                os.path.dirname(
                    os.path.abspath(output_path)
                ),
                exist_ok=True,
            )

            sf.write(
                output_path,
                audio,
                sample_rate,
            )

            return ModelResult(
                success=True,
                model=self.name,
                data=output_path,
                metadata={
                    "type": "music",
                    "path": output_path,
                    "sample_rate": sample_rate,
                    "local": True,
                    "offline": True,
                    "external_api": False,
                },
            )

        except Exception as error:

            return ModelResult(
                success=False,
                model=self.name,
                error=(
                    f"{error.__class__.__name__}: "
                    f"{error}"
                ),
            )

    def run(self, **kwargs) -> ModelResult:

        prompt = kwargs.get(
            "prompt",
            kwargs.get(
                "text",
                kwargs.get("input"),
            ),
        )

        if prompt is None:
            return ModelResult(
                success=False,
                model=self.name,
                error="Music prompt is required.",
            )

        return self.generate_music(
            prompt=str(prompt),
            output_path=kwargs.get(
                "output_path"
            ),
            max_new_tokens=int(
                kwargs.get(
                    "max_new_tokens",
                    256,
                )
            ),
        )


# =========================================================
# DIFFUSION MODEL BASE
# =========================================================

class DiffusionModelAdapter(ModelAdapter):

    def _load_diffusers_pipeline(
        self,
        pipeline_class_name: Optional[str] = None,
    ):

        model_path = self.ensure_local()

        try:
            import torch
            import diffusers

        except Exception as error:
            raise RuntimeError(
                "Diffusers/PyTorch is required: "
                f"{error}"
            ) from error

        dtype = (
            torch.float16
            if USE_GPU
            else torch.float32
        )

        if pipeline_class_name:

            pipeline_class = getattr(
                diffusers,
                pipeline_class_name,
                None,
            )

            if pipeline_class is None:
                raise RuntimeError(
                    f"Diffusers does not provide "
                    f"{pipeline_class_name}."
                )

            self.pipeline = (
                pipeline_class.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            )

        else:

            self.pipeline = (
                diffusers.DiffusionPipeline
                .from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            )

        device = (
            "cuda"
            if USE_GPU
            else "cpu"
        )

        self.pipeline = (
            self.pipeline.to(device)
        )

        self.loaded = True


# =========================================================
# WAN2.1 — VIDEO
# =========================================================

class WanVideoModelAdapter(
    DiffusionModelAdapter
):

    def load(self) -> None:

        with self.lock:

            if self.loaded:
                return

            self._load_diffusers_pipeline()

    def generate_video(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        num_frames: int = 16,
        height: int = 480,
        width: int = 832,
    ) -> ModelResult:

        try:

            self.load()

            if self.pipeline is None:
                raise RuntimeError(
                    "Wan pipeline is not loaded."
                )

            if output_path is None:
                output_path = os.path.join(
                    os.getcwd(),
                    "iraai_video.mp4",
                )

            result = self.pipeline(
                prompt=str(prompt),
                num_frames=max(
                    1,
                    int(num_frames),
                ),
                height=max(
                    64,
                    int(height),
                ),
                width=max(
                    64,
                    int(width),
                ),
            )

            frames = getattr(
                result,
                "frames",
                None,
            )

            if frames is None:
                raise RuntimeError(
                    "Video pipeline returned no frames."
                )

            if (
                isinstance(frames, list)
                and frames
                and isinstance(frames[0], list)
            ):
                frames = frames[0]

            os.makedirs(
                os.path.dirname(
                    os.path.abspath(output_path)
                ),
                exist_ok=True,
            )

            try:
                import imageio.v2 as imageio

                imageio.mimsave(
                    output_path,
                    frames,
                    fps=8,
                )

            except Exception as error:
                raise RuntimeError(
                    "imageio is required to save "
                    f"the generated video: {error}"
                ) from error

            return ModelResult(
                success=True,
                model=self.name,
                data=output_path,
                metadata={
                    "type": "video",
                    "path": output_path,
                    "frames": len(frames),
                    "local": True,
                    "offline": True,
                    "external_api": False,
                },
            )

        except Exception as error:

            return ModelResult(
                success=False,
                model=self.name,
                error=(
                    f"{error.__class__.__name__}: "
                    f"{error}"
                ),
            )

    def run(self, **kwargs) -> ModelResult:

        prompt = kwargs.get(
            "prompt",
            kwargs.get(
                "text",
                kwargs.get("input"),
            ),
        )

        if prompt is None:
            return ModelResult(
                success=False,
                model=self.name,
                error="Video prompt is required.",
            )

        return self.generate_video(
            prompt=str(prompt),
            output_path=kwargs.get(
                "output_path"
            ),
            num_frames=int(
                kwargs.get(
                    "num_frames",
                    16,
                )
            ),
            height=int(
                kwargs.get(
                    "height",
                    480,
                )
            ),
            width=int(
                kwargs.get(
                    "width",
                    832,
                )
            ),
        )


# =========================================================
# SDXL BASE — IMAGE
# =========================================================

class SDXLBaseModelAdapter(
    DiffusionModelAdapter
):

    def load(self) -> None:

        with self.lock:

            if self.loaded:
                return

            self._load_diffusers_pipeline(
                "StableDiffusionXLPipeline"
            )

    def generate_image(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        guidance_scale: float = 5.0,
    ) -> ModelResult:

        try:

            self.load()

            if output_path is None:
                output_path = os.path.join(
                    os.getcwd(),
                    "iraai_image.png",
                )

            result = self.pipeline(
                prompt=str(prompt),
                width=max(
                    64,
                    int(width),
                ),
                height=max(
                    64,
                    int(height),
                ),
                num_inference_steps=max(
                    1,
                    int(steps),
                ),
                guidance_scale=float(
                    guidance_scale
                ),
            )

            image = result.images[0]

            os.makedirs(
                os.path.dirname(
                    os.path.abspath(output_path)
                ),
                exist_ok=True,
            )

            image.save(
                output_path
            )

            return ModelResult(
                success=True,
                model=self.name,
                data=output_path,
                metadata={
                    "type": "image",
                    "path": output_path,
                    "width": image.width,
                    "height": image.height,
                    "local": True,
                    "offline": True,
                    "external_api": False,
                },
            )

        except Exception as error:

            return ModelResult(
                success=False,
                model=self.name,
                error=(
                    f"{error.__class__.__name__}: "
                    f"{error}"
                ),
            )

    def run(self, **kwargs) -> ModelResult:

        prompt = kwargs.get(
            "prompt",
            kwargs.get(
                "text",
                kwargs.get("input"),
            ),
        )

        if prompt is None:
            return ModelResult(
                success=False,
                model=self.name,
                error="Image prompt is required.",
            )

        return self.generate_image(
            prompt=str(prompt),
            output_path=kwargs.get(
                "output_path"
            ),
            width=int(
                kwargs.get(
                    "width",
                    1024,
                )
            ),
            height=int(
                kwargs.get(
                    "height",
                    1024,
                )
            ),
            steps=int(
                kwargs.get(
                    "steps",
                    30,
                )
            ),
            guidance_scale=float(
                kwargs.get(
                    "guidance_scale",
                    5.0,
                )
            ),
        )


# =========================================================
# SDXL REFINER
# =========================================================

class SDXLRefinerModelAdapter(
    DiffusionModelAdapter
):

    def load(self) -> None:

        with self.lock:

            if self.loaded:
                return

            self._load_diffusers_pipeline(
                "StableDiffusionXLImg2ImgPipeline"
            )

    def refine(
        self,
        image: Any,
        prompt: str = "",
        output_path: Optional[str] = None,
        strength: float = 0.3,
        steps: int = 30,
    ) -> ModelResult:

        try:

            self.load()

            if isinstance(image, str):

                if not os.path.isfile(image):
                    raise FileNotFoundError(
                        f"Image file not found: {image}"
                    )

                from PIL import Image

                image = Image.open(
                    image
                ).convert("RGB")

            if image is None:
                raise ValueError(
                    "Image input is required."
                )

            if output_path is None:
                output_path = os.path.join(
                    os.getcwd(),
                    "iraai_refined.png",
                )

            result = self.pipeline(
                prompt=str(prompt),
                image=image,
                strength=min(
                    1.0,
                    max(
                        0.0,
                        float(strength),
                    ),
                ),
                num_inference_steps=max(
                    1,
                    int(steps),
                ),
            )

            refined = result.images[0]

            os.makedirs(
                os.path.dirname(
                    os.path.abspath(output_path)
                ),
                exist_ok=True,
            )

            refined.save(
                output_path
            )

            return ModelResult(
                success=True,
                model=self.name,
                data=output_path,
                metadata={
                    "type": "image_refiner",
                    "path": output_path,
                    "local": True,
                    "offline": True,
                    "external_api": False,
                },
            )

        except Exception as error:

            return ModelResult(
                success=False,
                model=self.name,
                error=(
                    f"{error.__class__.__name__}: "
                    f"{error}"
                ),
            )

    def run(self, **kwargs) -> ModelResult:

        image = kwargs.get(
            "image",
            kwargs.get("input"),
        )

        if image is None:
            return ModelResult(
                success=False,
                model=self.name,
                error="Image input is required.",
            )

        return self.refine(
            image=image,
            prompt=str(
                kwargs.get(
                    "prompt",
                    "",
                )
            ),
            output_path=kwargs.get(
                "output_path"
            ),
            strength=float(
                kwargs.get(
                    "strength",
                    0.3,
                )
            ),
            steps=int(
                kwargs.get(
                    "steps",
                    30,
                )
            ),
        )


# =========================================================
# BGE-M3 — EMBEDDING / MEMORY
# =========================================================

class BGEEmbeddingModelAdapter(
    ModelAdapter
):

    def load(self) -> None:

        with self.lock:

            if self.loaded:
                return

            model_path = self.ensure_local()

            try:
                import torch

                from transformers import (
                    AutoModel,
                    AutoTokenizer,
                )

            except Exception as error:
                raise RuntimeError(
                    "BGE-M3 dependencies are "
                    f"not available: {error}"
                ) from error

            self.tokenizer = (
                AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=TRUST_REMOTE_CODE,
                )
            )

            dtype = (
                torch.float16
                if USE_GPU
                else torch.float32
            )

            kwargs: Dict[str, Any] = {
                "local_files_only": True,
                "trust_remote_code": TRUST_REMOTE_CODE,
                "torch_dtype": dtype,
            }

            if USE_GPU:
                kwargs["device_map"] = "auto"

            self.model = (
                AutoModel.from_pretrained(
                    model_path,
                    **kwargs,
                )
            )

            if not USE_GPU:
                self.model.to("cpu")

            self.model.eval()
            self.loaded = True

    # =====================================================
    # MEAN POOLING
    # =====================================================

    @staticmethod
    def _mean_pooling(
        model_output,
        attention_mask,
    ):

        import torch

        token_embeddings = model_output[0]

        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(
                token_embeddings.size()
            )
            .float()
        )

        return torch.sum(
            token_embeddings * mask,
            dim=1,
        ) / torch.clamp(
            mask.sum(dim=1),
            min=1e-9,
        )

    # =====================================================
    # EMBED
    # =====================================================

    def embed(
        self,
        text: Any,
    ) -> ModelResult:

        try:

            self.load()

            if isinstance(text, str):
                texts = [text]
            else:
                texts = [
                    str(item)
                    for item in text
                ]

            if not texts:
                raise ValueError(
                    "Text input is empty."
                )

            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            try:
                device = next(
                    self.model.parameters()
                ).device

            except StopIteration:
                device = (
                    "cuda"
                    if USE_GPU
                    else "cpu"
                )

            inputs = {
                key: value.to(device)
                if hasattr(value, "to")
                else value
                for key, value in inputs.items()
            }

            import torch
            import torch.nn.functional as F

            with torch.inference_mode():

                outputs = self.model(
                    **inputs
                )

                embeddings = (
                    self._mean_pooling(
                        outputs,
                        inputs["attention_mask"],
                    )
                )

                embeddings = F.normalize(
                    embeddings,
                    p=2,
                    dim=1,
                )

            vectors = (
                embeddings
                .detach()
                .float()
                .cpu()
                .tolist()
            )

            data = (
                vectors[0]
                if len(vectors) == 1
                else vectors
            )

            return ModelResult(
                success=True,
                model=self.name,
                data=data,
                metadata={
                    "type": "embedding",
                    "dimensions": len(
                        vectors[0]
                    ),
                    "count": len(vectors),
                    "local": True,
                    "offline": True,
                    "external_api": False,
                },
            )

        except Exception as error:

            return ModelResult(
                success=False,
                model=self.name,
                error=(
                    f"{error.__class__.__name__}: "
                    f"{error}"
                ),
            )

    def run(self, **kwargs) -> ModelResult:

        text = kwargs.get(
            "text",
            kwargs.get("input"),
        )

        if text is None:
            return ModelResult(
                success=False,
                model=self.name,
                error="Text input is required.",
            )

        return self.embed(text)


# =========================================================
# MODEL MANAGER
# =========================================================

class ModelManager:

    def __init__(self):

        self.models: Dict[
            str,
            ModelAdapter
        ] = {}

        self.lock = threading.RLock()

        self._register_configured_models()

    # =====================================================
    # REGISTER
    # =====================================================

    def register_model(
        self,
        name: str,
        config: Dict[str, Any],
    ) -> None:

        if not name:
            raise ValueError(
                "Model name cannot be empty."
            )

        model_type = str(
            config.get(
                "type",
                "",
            )
        ).lower()

        category = str(
            config.get(
                "category",
                "",
            )
        ).lower()

        combined = (
            f"{model_type} "
            f"{category} "
            f"{name}"
        ).lower()

        if (
            "vision" in combined
            or "qwen3-vl" in combined
        ):

            adapter = VisionModelAdapter(
                name=name,
                config=config,
            )

        elif (
            "whisper" in combined
            or "speech_to_text" in combined
        ):

            adapter = WhisperModelAdapter(
                name=name,
                config=config,
            )

        elif (
            "kokoro" in combined
            or "text_to_speech" in combined
        ):

            adapter = KokoroModelAdapter(
                name=name,
                config=config,
            )

        elif (
            "musicgen" in combined
            or model_type == "music"
        ):

            adapter = MusicGenModelAdapter(
                name=name,
                config=config,
            )

        elif (
            "wan2.1" in combined
            or model_type == "video"
        ):

            adapter = WanVideoModelAdapter(
                name=name,
                config=config,
            )

        elif (
            "stable-diffusion-xl-refiner" in combined
            or model_type == "image_refiner"
        ):

            adapter = SDXLRefinerModelAdapter(
                name=name,
                config=config,
            )

        elif (
            "stable-diffusion-xl-base" in combined
            or model_type == "image"
        ):

            adapter = SDXLBaseModelAdapter(
                name=name,
                config=config,
            )

        elif (
            "bge-m3" in combined
            or model_type == "embedding"
        ):

            adapter = BGEEmbeddingModelAdapter(
                name=name,
                config=config,
            )

        elif model_type in {
            "general",
            "coder",
            "reasoning",
            "text",
            "language",
        }:

            adapter = TextModelAdapter(
                name=name,
                config=config,
            )

        else:

            role = self._find_role_for_model(
                name
            )

            if role == "vision":

                adapter = VisionModelAdapter(
                    name=name,
                    config=config,
                )

            elif role == "speech_to_text":

                adapter = WhisperModelAdapter(
                    name=name,
                    config=config,
                )

            elif role == "text_to_speech":

                adapter = KokoroModelAdapter(
                    name=name,
                    config=config,
                )

            elif role == "music":

                adapter = MusicGenModelAdapter(
                    name=name,
                    config=config,
                )

            elif role == "video":

                adapter = WanVideoModelAdapter(
                    name=name,
                    config=config,
                )

            elif role == "image":

                adapter = SDXLBaseModelAdapter(
                    name=name,
                    config=config,
                )

            elif role == "image_refiner":

                adapter = SDXLRefinerModelAdapter(
                    name=name,
                    config=config,
                )

            elif role == "embedding":

                adapter = BGEEmbeddingModelAdapter(
                    name=name,
                    config=config,
                )

            else:

                adapter = TextModelAdapter(
                    name=name,
                    config=config,
                )

        with self.lock:
            self.models[name] = adapter

    # =====================================================
    # FIND ROLE
    # =====================================================

    def _find_role_for_model(
        self,
        model_name: str,
    ) -> Optional[str]:

        for role, selected in MODEL_ROLES.items():

            if selected == model_name:
                return role

        return None

    # =====================================================
    # REGISTER CONFIGURED MODELS
    # =====================================================

    def _register_configured_models(
        self,
    ) -> None:

        for name, config in LOCAL_MODELS.items():

            self.register_model(
                name=name,
                config=config,
            )

    # =====================================================
    # LIST
    # =====================================================

    def get_models(self) -> List[str]:

        with self.lock:
            return list(
                self.models.keys()
            )

    # =====================================================
    # GET ALL MODELS
    # =====================================================

    def get_all_models(
        self,
    ) -> Dict[str, Dict[str, Any]]:

        with self.lock:

            return {
                name: adapter.health()
                for name, adapter
                in self.models.items()
            }

    # =====================================================
    # GET MODEL
    # =====================================================

    def get_model(
        self,
        name: Optional[str] = None,
    ) -> Optional[ModelAdapter]:

        selected = (
            name
            or DEFAULT_MODEL
        )

        if selected in self.models:
            return self.models[selected]

        normalized = str(
            selected
        ).strip().lower()

        for model_name, adapter in (
            self.models.items()
        ):

            if model_name.lower() == normalized:
                return adapter

        role_model = MODEL_ROLES.get(
            normalized
        )

        if role_model in self.models:
            return self.models[
                role_model
            ]

        return None

    # =====================================================
    # ROLE → MODEL
    # =====================================================

    def resolve_role(
        self,
        role: str,
    ) -> Optional[str]:

        if not role:
            return None

        normalized = (
            str(role)
            .strip()
            .lower()
        )

        selected = MODEL_ROLES.get(
            normalized
        )

        if selected in self.models:
            return selected

        return None

    # =====================================================
    # BUILD TEXT PROMPT
    # =====================================================

    def _build_prompt(
        self,
        prompt: str,
        context: Optional[str] = None,
    ) -> str:

        sections: List[str] = []

        instructions = (
            get_model_instructions()
        )

        if instructions:

            sections.append(
                "SYSTEM INSTRUCTIONS:\n"
                + instructions
            )

        if context:

            sections.append(
                "CONTEXT:\n"
                + str(context)
            )

        sections.append(
            "USER REQUEST:\n"
            + str(prompt)
        )

        sections.append(
            "ASSISTANT:"
        )

        return "\n\n".join(
            sections
        )

    # =====================================================
    # GENERATE TEXT
    # =====================================================

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        role: Optional[str] = None,
        context: Optional[str] = None,
        temperature: float = AI_TEMPERATURE,
        max_new_tokens: int = AI_MAX_NEW_TOKENS,
        top_p: float = AI_TOP_P,
        do_sample: bool = AI_DO_SAMPLE,
    ) -> ModelResult:

        selected_model = model

        if not selected_model and role:

            selected_model = (
                self.resolve_role(role)
            )

        if not selected_model:
            selected_model = DEFAULT_MODEL

        adapter = self.get_model(
            selected_model
        )

        if adapter is None:

            return ModelResult(
                success=False,
                model=selected_model,
                error=(
                    "Model is not registered: "
                    f"{selected_model}"
                ),
            )

        if not isinstance(
            adapter,
            TextModelAdapter,
        ):

            return ModelResult(
                success=False,
                model=selected_model,
                error=(
                    "Selected model is not a "
                    "text-generation model."
                ),
                metadata={
                    "type": adapter.model_type,
                },
            )

        full_prompt = self._build_prompt(
            prompt=prompt,
            context=context,
        )

        try:

            text = adapter.generate(
                prompt=full_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                do_sample=do_sample,
            )

            return ModelResult(
                success=True,
                text=text,
                model=selected_model,
                metadata={
                    "type": adapter.model_type,
                    "category": adapter.category,
                    "repository": adapter.repository,
                    "local": True,
                    "offline": True,
                    "external_api": False,
                },
            )

        except Exception as error:

            return ModelResult(
                success=False,
                model=selected_model,
                error=(
                    f"{error.__class__.__name__}: "
                    f"{error}"
                ),
                metadata={
                    "type": adapter.model_type,
                    "repository": adapter.repository,
                    "local": True,
                    "offline": True,
                    "external_api": False,
                },
            )

    # =====================================================
    # CHAT
    # =====================================================

    def chat(
        self,
        message: str,
        context: Optional[str] = None,
        model: Optional[str] = None,
        role: Optional[str] = None,
    ) -> ModelResult:

        return self.generate(
            prompt=message,
            model=model,
            role=role,
            context=context,
        )

    # =====================================================
    # SPECIALIZED
    # =====================================================

    def run_specialized(
        self,
        model: str,
        **kwargs,
    ) -> ModelResult:

        adapter = self.get_model(
            model
        )

        if adapter is None:

            return ModelResult(
                success=False,
                model=model,
                error=(
                    "Model is not registered."
                ),
            )

        if isinstance(
            adapter,
            TextModelAdapter,
        ):

            return ModelResult(
                success=False,
                model=model,
                error=(
                    "This model is a text model. "
                    "Use generate() instead."
                ),
            )

        if isinstance(
            adapter,
            VisionModelAdapter,
        ):

            return adapter.generate(
                prompt=str(
                    kwargs.get(
                        "prompt",
                        "",
                    )
                ),
                image=kwargs.get(
                    "image"
                ),
                max_new_tokens=int(
                    kwargs.get(
                        "max_new_tokens",
                        AI_MAX_NEW_TOKENS,
                    )
                ),
            )

        if hasattr(
            adapter,
            "run"
        ):

            return adapter.run(
                **kwargs
            )

        return ModelResult(
            success=False,
            model=model,
            error=(
                "No inference operation is "
                "available for this model."
            ),
        )

    # =====================================================
    # VISION
    # =====================================================

    def vision(
        self,
        image: Any,
        prompt: str = "",
    ) -> ModelResult:

        model_name = self.resolve_role(
            "vision"
        )

        if not model_name:

            return ModelResult(
                success=False,
                error=(
                    "Vision model is not "
                    "configured."
                ),
            )

        adapter = self.get_model(
            model_name
        )

        if not isinstance(
            adapter,
            VisionModelAdapter,
        ):

            return ModelResult(
                success=False,
                model=model_name,
                error=(
                    "Vision model is not "
                    "configured correctly."
                ),
            )

        vision_prompt = self._build_prompt(
            prompt=prompt
        )

        return adapter.generate(
            prompt=vision_prompt,
            image=image,
        )

    # =====================================================
    # UNLOAD ONE
    # =====================================================

    def unload_model(
        self,
        name: str,
    ) -> bool:

        adapter = self.get_model(
            name
        )

        if adapter is None:
            return False

        adapter.unload()

        return True

    # =====================================================
    # UNLOAD ALL
    # =====================================================

    def unload_all(self) -> None:

        with self.lock:

            adapters = list(
                self.models.values()
            )

        for adapter in adapters:
            adapter.unload()

    # =====================================================
    # HEALTH
    # =====================================================

    def health(
        self,
    ) -> Dict[str, Any]:

        with self.lock:

            model_status = {
                name: adapter.health()
                for name, adapter
                in self.models.items()
            }

        return {
            "available": bool(model_status),
            "local": True,
            "offline": True,
            "external_api": False,
            "device": DEVICE,
            "default_model": DEFAULT_MODEL,
            "model_count": len(
                model_status
            ),
            "models": model_status,
            "roles": MODEL_ROLES,
            "sources": HF_REPOSITORIES,
        }


# =========================================================
# GLOBAL MODEL MANAGER
# =========================================================

_model_manager: Optional[
    ModelManager
] = None

_model_manager_lock = (
    threading.RLock()
)


def get_model_manager() -> ModelManager:

    global _model_manager

    with _model_manager_lock:

        if _model_manager is None:

            _model_manager = (
                ModelManager()
            )

        return _model_manager


# =========================================================
# PUBLIC TEXT GENERATION
# =========================================================

def generate_model_response(
    prompt: str,
    model: Optional[str] = None,
    context: Optional[str] = None,
) -> ModelResult:

    return (
        get_model_manager()
        .generate(
            prompt=prompt,
            model=model,
            context=context,
        )
    )


# =========================================================
# PUBLIC ROLE GENERATION
# =========================================================

def generate_role_response(
    prompt: str,
    role: str,
    context: Optional[str] = None,
) -> ModelResult:

    return (
        get_model_manager()
        .generate(
            prompt=prompt,
            role=role,
            context=context,
        )
    )


# =========================================================
# PUBLIC VISION
# =========================================================

def analyze_image(
    image: Any,
    prompt: str = "",
) -> ModelResult:

    return (
        get_model_manager()
        .vision(
            image=image,
            prompt=prompt,
        )
    )


# =========================================================
# PUBLIC SPECIALIZED
# =========================================================

def run_model(
    model: str,
    **kwargs,
) -> ModelResult:

    return (
        get_model_manager()
        .run_specialized(
            model=model,
            **kwargs,
        )
    )


# =========================================================
# PUBLIC HEALTH
# =========================================================

def model_health() -> Dict[str, Any]:

    return (
        get_model_manager()
        .health()
    )


# =========================================================
# PUBLIC MODEL LIST
# =========================================================

def get_available_models() -> List[str]:

    return (
        get_model_manager()
        .get_models()
    )


# =========================================================
# PUBLIC MODEL DETAILS
# =========================================================

def get_all_models() -> Dict[str, Dict[str, Any]]:

    return (
        get_model_manager()
        .get_all_models()
    )


# =========================================================
# END
# =========================================================