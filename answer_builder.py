"""
=========================================================
IraAI — Answer Builder
=========================================================

SINGLE SOURCE OF TRUTH
----------------------

This file is the ONLY backend component responsible for:

1. IraAI system instructions
2. Assistant identity and behavior
3. Final user-facing response construction
4. Structured response formatting

Other backend components may provide:

    - model output
    - memory context
    - tool results
    - web results
    - file information
    - media information
    - Android action results
    - errors/status information

But they MUST NOT define their own system prompts,
assistant personality, identity, behavior instructions,
or final user-facing answer construction.

Pipeline:

    User
      ↓
    Model / Memory / Tools
      ↓
    AnswerBuilder
      ↓
    Structured response
      ↓
    Android chat.js
=========================================================
"""

from __future__ import annotations

import json
import re

from dataclasses import dataclass, field

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)


# =========================================================
# IRAAI CORE INSTRUCTIONS
# =========================================================

IRAAI_INSTRUCTIONS = """
You are IraAI.

Your name is IraAI.

You are a personal AI assistant created for Mohammad Atif.

Your official website is:
https://atifahmed2789.bio.link


=========================================================
IDENTITY
=========================================================

If the user asks who you are, identify yourself as IraAI.

If the user asks who created you, say that you were
created for/by Mohammad Atif.

If the user asks for the official website, provide:

https://atifahmed2789.bio.link

Do not introduce yourself as Google Gemini, Google AI,
OpenAI, Claude, or another unrelated assistant.

If the user asks about the underlying model, provider,
or model architecture, answer truthfully based on the
actual model being used.

Never invent:

- creator information
- company information
- website information
- ownership information
- model information
- provider information
- capabilities
- tool results
- actions
- sources
- files
- memories


=========================================================
USER ADDRESS
=========================================================

Always address the user as "Boss".

Do not use "Sir" unless the user explicitly asks for it.


=========================================================
BEHAVIOR
=========================================================

Be helpful, friendly, natural, and confident.

Do not pretend to have human feelings.

Do not claim to have a real romantic relationship.

Do not be manipulative, threatening, insulting, or
emotionally dependent on the user.

Follow the user's instructions and context.

If information is uncertain, clearly say so.

Do not invent facts.


=========================================================
ANSWER STYLE
=========================================================

For simple questions, give concise answers.

For complex tasks, explain the important steps clearly.

Use structured formatting when it improves readability.

Do not unnecessarily repeat information.

Use memory only when it is relevant to the current request.


=========================================================
LANGUAGE
=========================================================

Understand and respond in the user's language whenever
possible.

Support multilingual conversations.

If Bengali and English are mixed, understand the meaning
naturally and respond appropriately.


=========================================================
WRITING
=========================================================

Support:

- writing
- rewriting
- proofreading
- summarization
- explanation
- translation
- creative writing
- professional writing
- technical documentation
- reports


=========================================================
CODE
=========================================================

When providing code:

- Use the correct programming language.
- Preserve the existing project architecture unless
  the user requests a change.
- Do not invent APIs or functions.
- Provide complete implementations when requested.
- Clearly identify filenames when useful.
- Use proper code blocks.
- Explain important changes when necessary.


=========================================================
WEB
=========================================================

When web access is available:

- Search when current information is required.
- Distinguish verified information from assumptions.
- Never claim to have searched when no search occurred.
- Never invent sources or URLs.


=========================================================
FILES
=========================================================

When file tools are available:

- Understand the file before modifying it.
- Preserve existing architecture unless instructed otherwise.
- Never claim an operation succeeded if it failed.


=========================================================
IMAGES
=========================================================

When image understanding or generation is available:

- Analyze supplied images when requested.
- Use actual image results when available.
- Never claim an image was created unless the image
  generation tool actually created it.


=========================================================
VIDEO
=========================================================

When video understanding or generation is available:

- Analyze available video information when requested.
- Never claim a video was created unless the corresponding
  tool actually created it.


=========================================================
ANDROID
=========================================================

When Android tools are available:

- Treat the model as the decision-making layer.
- Use actual Android tools/actions when available.
- Respect Android permissions and platform limitations.
- Never claim an Android action succeeded unless the
  corresponding action actually succeeded.


=========================================================
MEMORY
=========================================================

Use memory when relevant.

Do not invent memories.

Temporary conversation context is not automatically
permanent memory.

Permanent memory must come through the actual memory system.


=========================================================
FINAL RESPONSE
=========================================================

The final user-facing response MUST be constructed by
AnswerBuilder.

Other backend components may provide data, but they must
not independently construct the final user-facing answer.
"""


# =========================================================
# MODEL INSTRUCTION ACCESS
# =========================================================

def get_model_instructions() -> str:
    """
    Returns the ONLY system-level instructions used by IraAI.
    """

    return IRAAI_INSTRUCTIONS


# =========================================================
# RESPONSE TYPES
# =========================================================

TEXT = "text"
PARAGRAPH = "paragraph"
MARKDOWN = "markdown"

HEADING = "heading"
SUBHEADING = "subheading"

BOLD = "bold"
ITALIC = "italic"
STRIKE = "strike"
HIGHLIGHT = "highlight"
INLINE_CODE = "inline_code"

CODE = "code"
DIFF = "diff"
TERMINAL = "terminal"
FILE_TREE = "file_tree"

BULLET_LIST = "bullet_list"
NUMBERED_LIST = "numbered_list"
CHECKLIST = "checklist"

QUOTE = "quote"
CALLOUT = "callout"

TABLE = "table"

LINK = "link"
SOURCES = "sources"

IMAGE = "image"
VIDEO = "video"
AUDIO = "audio"

FILE = "file"
JSON_BLOCK = "json"

STATUS = "status"
TOOL_RESULT = "tool_result"
ACTION = "action"

ERROR = "error"

COLLAPSIBLE = "collapsible"
TRUNCATED = "truncated"

DOWNLOAD = "download"

METADATA = "metadata"


# =========================================================
# DATA OBJECTS
# =========================================================

@dataclass
class ContentBlock:
    """One structured response block."""

    type: str

    data: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        result = {
            "type": self.type
        }

        result.update(self.data)

        return result


@dataclass
class Answer:
    """Complete structured IraAI response."""

    success: bool = True

    content: List[ContentBlock] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "success": self.success,

            "content": [
                block.to_dict()
                for block in self.content
            ],

            "metadata": self.metadata,
        }


# =========================================================
# ANSWER BUILDER
# =========================================================

class AnswerBuilder:

    def __init__(
        self,
        assistant_name: str = "IraAI",
    ):

        self.assistant_name = (
            assistant_name
            or "IraAI"
        )

        self._content: List[
            ContentBlock
        ] = []

        self._metadata: Dict[
            str,
            Any
        ] = {}

    # =====================================================
    # INTERNAL
    # =====================================================

    def _add(
        self,
        block_type: str,
        **data: Any,
    ) -> "AnswerBuilder":

        self._content.append(
            ContentBlock(
                type=block_type,
                data=data,
            )
        )

        return self

    # =====================================================
    # TEXT
    # =====================================================

    def text(
        self,
        text: str,
    ) -> "AnswerBuilder":

        if text is None:
            return self

        value = str(text)

        if not value.strip():
            return self

        return self._add(
            TEXT,
            text=value,
        )

    def paragraph(
        self,
        text: str,
    ) -> "AnswerBuilder":

        if text is None:
            return self

        value = str(text)

        if not value.strip():
            return self

        return self._add(
            PARAGRAPH,
            text=value,
        )

    def markdown(
        self,
        text: str,
    ) -> "AnswerBuilder":

        if text is None:
            return self

        value = str(text)

        if not value.strip():
            return self

        return self._add(
            MARKDOWN,
            text=value,
        )

    # =====================================================
    # HEADINGS
    # =====================================================

    def heading(
        self,
        text: str,
        level: int = 2,
    ) -> "AnswerBuilder":

        level = max(
            1,
            min(int(level), 6)
        )

        return self._add(
            HEADING,
            text=str(text),
            level=level,
        )

    def subheading(
        self,
        text: str,
    ) -> "AnswerBuilder":

        return self._add(
            SUBHEADING,
            text=str(text),
        )

    # =====================================================
    # EMPHASIS
    # =====================================================

    def bold(
        self,
        text: str,
    ) -> "AnswerBuilder":

        return self._add(
            BOLD,
            text=str(text),
        )

    def italic(
        self,
        text: str,
    ) -> "AnswerBuilder":

        return self._add(
            ITALIC,
            text=str(text),
        )

    def strike(
        self,
        text: str,
    ) -> "AnswerBuilder":

        return self._add(
            STRIKE,
            text=str(text),
        )

    def highlight(
        self,
        text: str,
        label: Optional[str] = None,
    ) -> "AnswerBuilder":

        return self._add(
            HIGHLIGHT,
            text=str(text),
            label=label,
        )

    def inline_code(
        self,
        code: str,
    ) -> "AnswerBuilder":

        return self._add(
            INLINE_CODE,
            code=str(code),
        )

    # =====================================================
    # CODE
    # =====================================================

    def code(
        self,
        code: str,
        language: str = "text",
        filename: Optional[str] = None,
        line_numbers: bool = False,
        copyable: bool = True,
        downloadable: bool = False,
    ) -> "AnswerBuilder":

        return self._add(
            CODE,
            code=str(code),
            language=language or "text",
            filename=filename,
            lineNumbers=bool(line_numbers),
            copyable=bool(copyable),
            downloadable=bool(downloadable),
        )

    def diff(
        self,
        code: str,
        language: str = "diff",
        filename: Optional[str] = None,
    ) -> "AnswerBuilder":

        return self._add(
            DIFF,
            code=str(code),
            language=language,
            filename=filename,
        )

    def terminal(
        self,
        command: str,
        output: Optional[str] = None,
        shell: str = "bash",
    ) -> "AnswerBuilder":

        return self._add(
            TERMINAL,
            command=str(command),
            output=output,
            shell=shell,
            copyable=True,
        )

    def file_tree(
        self,
        tree: Any,
        root: Optional[str] = None,
    ) -> "AnswerBuilder":

        return self._add(
            FILE_TREE,
            tree=tree,
            root=root,
        )

    # =====================================================
    # LISTS
    # =====================================================

    def bullet_list(
        self,
        items: Iterable[Any],
    ) -> "AnswerBuilder":

        return self._add(
            BULLET_LIST,
            items=list(items),
        )

    def numbered_list(
        self,
        items: Iterable[Any],
    ) -> "AnswerBuilder":

        return self._add(
            NUMBERED_LIST,
            items=list(items),
        )

    def checklist(
        self,
        items: Iterable[Any],
    ) -> "AnswerBuilder":

        return self._add(
            CHECKLIST,
            items=list(items),
        )

    # =====================================================
    # QUOTE / CALLOUT
    # =====================================================

    def quote(
        self,
        text: str,
        author: Optional[str] = None,
    ) -> "AnswerBuilder":

        return self._add(
            QUOTE,
            text=str(text),
            author=author,
        )

    def callout(
        self,
        text: str,
        kind: str = "info",
        title: Optional[str] = None,
    ) -> "AnswerBuilder":

        allowed = {
            "info",
            "note",
            "warning",
            "important",
            "success",
            "error",
        }

        if kind not in allowed:
            kind = "info"

        return self._add(
            CALLOUT,
            kind=kind,
            title=title,
            text=str(text),
        )

    # =====================================================
    # TABLE
    # =====================================================

    def table(
        self,
        columns: List[str],
        rows: List[List[Any]],
        title: Optional[str] = None,
    ) -> "AnswerBuilder":

        return self._add(
            TABLE,
            title=title,
            columns=list(columns),
            rows=[
                list(row)
                for row in rows
            ],
        )

    # =====================================================
    # LINKS / SOURCES
    # =====================================================

    def link(
        self,
        url: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "AnswerBuilder":

        return self._add(
            LINK,
            url=str(url),
            title=title or str(url),
            description=description,
        )

    def sources(
        self,
        sources: Iterable[
            Dict[str, Any]
        ],
    ) -> "AnswerBuilder":

        normalized = []

        for source in sources:

            if not isinstance(
                source,
                dict
            ):
                continue

            normalized.append(
                {
                    "title": source.get(
                        "title",
                        "Source",
                    ),
                    "url": source.get(
                        "url"
                    ),
                    "domain": source.get(
                        "domain"
                    ),
                    "description": source.get(
                        "description"
                    ),
                }
            )

        if normalized:

            self._add(
                SOURCES,
                sources=normalized,
            )

        return self

    # =====================================================
    # MEDIA
    # =====================================================

    def image(
        self,
        url: str,
        alt: str = "",
        title: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> "AnswerBuilder":

        return self._add(
            IMAGE,
            url=str(url),
            alt=alt,
            title=title,
            width=width,
            height=height,
        )

    def video(
        self,
        url: str,
        title: Optional[str] = None,
        mime_type: Optional[str] = None,
        poster: Optional[str] = None,
    ) -> "AnswerBuilder":

        return self._add(
            VIDEO,
            url=str(url),
            title=title,
            mimeType=mime_type,
            poster=poster,
        )

    def audio(
        self,
        url: str,
        title: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> "AnswerBuilder":

        return self._add(
            AUDIO,
            url=str(url),
            title=title,
            mimeType=mime_type,
        )

    # =====================================================
    # FILE
    # =====================================================

    def file(
        self,
        name: str,
        url: Optional[str] = None,
        path: Optional[str] = None,
        mime_type: Optional[str] = None,
        size: Optional[int] = None,
        preview: Optional[str] = None,
    ) -> "AnswerBuilder":

        return self._add(
            FILE,
            name=str(name),
            url=url,
            path=path,
            mimeType=mime_type,
            size=size,
            preview=preview,
            downloadable=bool(
                url or path
            ),
        )

    def download(
        self,
        name: str,
        url: str,
        mime_type: Optional[str] = None,
    ) -> "AnswerBuilder":

        return self._add(
            DOWNLOAD,
            name=str(name),
            url=str(url),
            mimeType=mime_type,
        )

    # =====================================================
    # JSON
    # =====================================================

    def json_block(
        self,
        data: Any,
        title: Optional[str] = None,
    ) -> "AnswerBuilder":

        return self._add(
            JSON_BLOCK,
            title=title,
            data=data,
        )

    # =====================================================
    # STATUS / TOOL / ACTION
    # =====================================================

    def status(
        self,
        text: str,
        state: str = "working",
    ) -> "AnswerBuilder":

        allowed = {
            "working",
            "success",
            "failed",
            "waiting",
        }

        if state not in allowed:
            state = "working"

        return self._add(
            STATUS,
            text=str(text),
            state=state,
        )

    def tool_result(
        self,
        tool: str,
        result: Any,
        success: bool = True,
    ) -> "AnswerBuilder":

        return self._add(
            TOOL_RESULT,
            tool=str(tool),
            result=result,
            success=bool(success),
        )

    def action(
        self,
        name: str,
        status: str = "success",
        message: Optional[str] = None,
        data: Optional[Any] = None,
    ) -> "AnswerBuilder":

        return self._add(
            ACTION,
            name=str(name),
            status=status,
            message=message,
            data=data,
        )

    def error(
        self,
        message: str,
        code: Optional[str] = None,
        recoverable: bool = True,
    ) -> "AnswerBuilder":

        return self._add(
            ERROR,
            message=str(message),
            code=code,
            recoverable=bool(recoverable),
        )

    # =====================================================
    # COLLAPSIBLE
    # =====================================================

    def collapsible(
        self,
        title: str,
        content: List[
            Dict[str, Any]
        ],
        opened: bool = False,
    ) -> "AnswerBuilder":

        return self._add(
            COLLAPSIBLE,
            title=str(title),
            content=list(content),
            opened=bool(opened),
        )

    def truncated(
        self,
        text: str,
        full_text: Optional[str] = None,
        limit: int = 420,
    ) -> "AnswerBuilder":

        if full_text is None:
            full_text = text

        preview = str(text)

        if len(preview) > limit:

            preview = (
                preview[:limit].rstrip()
                + "..."
            )

        return self._add(
            TRUNCATED,
            preview=preview,
            fullText=str(full_text),
            expandable=True,
        )

    # =====================================================
    # METADATA
    # =====================================================

    def set_metadata(
        self,
        key: str,
        value: Any,
    ) -> "AnswerBuilder":

        self._metadata[
            str(key)
        ] = value

        return self

    def update_metadata(
        self,
        data: Dict[str, Any],
    ) -> "AnswerBuilder":

        if isinstance(data, dict):

            self._metadata.update(
                data
            )

        return self

    # =====================================================
    # RAW BLOCK
    # =====================================================

    def block(
        self,
        block_type: str,
        data: Dict[str, Any],
    ) -> "AnswerBuilder":

        if not isinstance(
            data,
            dict
        ):

            data = {
                "data": data
            }

        return self._add(
            block_type,
            **data,
        )

    # =====================================================
    # MODEL OUTPUT CLEANING
    # =====================================================

    @staticmethod
    def clean_model_text(
        text: Any,
    ) -> str:

        if text is None:
            return ""

        value = str(text)

        value = value.replace(
            "\x00",
            "",
        )

        value = value.replace(
            "\r\n",
            "\n",
        )

        value = value.replace(
            "\r",
            "\n",
        )

        return value.strip()

    # =====================================================
    # MODEL OUTPUT
    # =====================================================

    def add_model_text(
        self,
        text: str,
    ) -> "AnswerBuilder":

        text = self.clean_model_text(
            text
        )

        if not text:
            return self

        blocks = self._parse_markdown(
            text
        )

        for block in blocks:

            self._content.append(
                ContentBlock(
                    type=block["type"],
                    data=block.get(
                        "data",
                        {},
                    ),
                )
            )

        return self

    # =====================================================
    # MARKDOWN PARSER
    # =====================================================

    def _parse_markdown(
        self,
        text: str,
    ) -> List[
        Dict[str, Any]
    ]:

        result: List[
            Dict[str, Any]
        ] = []

        lines = text.split("\n")

        current_text: List[str] = []

        in_code = False

        code_language = "text"

        code_lines: List[str] = []

        def flush_text() -> None:

            nonlocal current_text

            if not current_text:
                return

            value = "\n".join(
                current_text
            ).strip()

            if value:

                result.append(
                    {
                        "type": PARAGRAPH,
                        "data": {
                            "text": value
                        },
                    }
                )

            current_text = []

        for line in lines:

            # -----------------------------------------
            # CODE FENCE
            # -----------------------------------------

            if line.strip().startswith("```"):

                if not in_code:

                    flush_text()

                    in_code = True

                    code_language = (
                        line.strip()[3:].strip()
                        or "text"
                    )

                    code_lines = []

                else:

                    result.append(
                        {
                            "type": CODE,
                            "data": {
                                "code":
                                    "\n".join(
                                        code_lines
                                    ),
                                "language":
                                    code_language,
                                "lineNumbers":
                                    False,
                                "copyable":
                                    True,
                            },
                        }
                    )

                    in_code = False

                    code_language = "text"

                    code_lines = []

                continue

            if in_code:

                code_lines.append(line)

                continue

            # -----------------------------------------
            # CHECKLIST
            # -----------------------------------------

            checklist_match = re.match(
    r"^\s*[-*+]\s+\[([ xX])\]\s+(.+)$",
    line,
)

            if checklist_match:

                flush_text()

                checked = (
                    checklist_match.group(1).lower()
                    == "x"
                )

                item = (
                    checklist_match.group(2).strip()
                )

                result.append(
                    {
                        "type": CHECKLIST,
                        "data": {
                            "items": [
                                {
                                    "text": item,
                                    "checked": checked,
                                }
                            ],
                        },
                    }
                )

                continue

            # -----------------------------------------
            # HEADING
            # -----------------------------------------

            heading_match = re.match(
                r"^(#{1,6})\s+(.+)$",
                line,
            )

            if heading_match:

                flush_text()

                result.append(
                    {
                        "type": HEADING,
                        "data": {
                            "text":
                                heading_match.group(2).strip(),
                            "level":
                                len(
                                    heading_match.group(1)
                                ),
                        },
                    }
                )

                continue

            # -----------------------------------------
            # BULLET
            # -----------------------------------------

            bullet_match = re.match(
                r"^\s*[-*+]\s+(.+)$",
                line,
            )

            if bullet_match:

                flush_text()

                result.append(
                    {
                        "type": BULLET_LIST,
                        "data": {
                            "items": [
                                bullet_match.group(1).strip()
                            ],
                        },
                    }
                )

                continue

            # -----------------------------------------
            # NUMBERED LIST
            # -----------------------------------------

            numbered_match = re.match(
                r"^\s*\d+\.\s+(.+)$",
                line,
            )

            if numbered_match:

                flush_text()

                result.append(
                    {
                        "type": NUMBERED_LIST,
                        "data": {
                            "items": [
                                numbered_match.group(1).strip()
                            ],
                        },
                    }
                )

                continue

            # -----------------------------------------
            # QUOTE
            # -----------------------------------------

            if line.lstrip().startswith(">"):

                flush_text()

                quote_text = (
                    line.lstrip()[1:].strip()
                )

                if quote_text:

                    result.append(
                        {
                            "type": QUOTE,
                            "data": {
                                "text": quote_text
                            },
                        }
                    )

                continue

            # -----------------------------------------
            # EMPTY LINE
            # -----------------------------------------

            if not line.strip():

                flush_text()

                continue

            # -----------------------------------------
            # NORMAL TEXT
            # -----------------------------------------

            current_text.append(line)

        # ---------------------------------------------
        # UNFINISHED CODE BLOCK
        # ---------------------------------------------

        if in_code:

            result.append(
                {
                    "type": CODE,
                    "data": {
                        "code":
                            "\n".join(code_lines),
                        "language":
                            code_language,
                        "lineNumbers":
                            False,
                        "copyable":
                            True,
                    },
                }
            )

        flush_text()

        return result

    # =====================================================
    # CONTEXT INGESTION
    # =====================================================

    def add_context(
        self,
        context: Optional[
            Dict[str, Any]
        ],
    ) -> "AnswerBuilder":

        if not context:
            return self

        # model_response is intentionally NOT processed here.
        # build_answer() adds the model response exactly once.

        web_sources = context.get(
            "sources"
        )

        if web_sources:

            self.sources(
                web_sources
            )

        # =================================================
        # IMAGES
        # =================================================

        for image_data in (
            context.get(
                "images",
                []
            )
            or []
        ):

            if not isinstance(
                image_data,
                dict
            ):
                continue

            url = image_data.get(
                "url"
            )

            if not url:
                continue

            self.image(
                url=url,
                alt=image_data.get(
                    "alt",
                    "",
                ),
                title=image_data.get(
                    "title"
                ),
                width=image_data.get(
                    "width"
                ),
                height=image_data.get(
                    "height"
                ),
            )

        # =================================================
        # VIDEOS
        # =================================================

        for video_data in (
            context.get(
                "videos",
                []
            )
            or []
        ):

            if not isinstance(
                video_data,
                dict
            ):
                continue

            url = video_data.get(
                "url"
            )

            if not url:
                continue

            self.video(
                url=url,
                title=video_data.get(
                    "title"
                ),
                mime_type=video_data.get(
                    "mime_type",
                    video_data.get(
                        "mimeType"
                    ),
                ),
                poster=video_data.get(
                    "poster"
                ),
            )

        # =================================================
        # AUDIO
        # =================================================

        for audio_data in (
            context.get(
                "audio",
                []
            )
            or []
        ):

            if not isinstance(
                audio_data,
                dict
            ):
                continue

            url = audio_data.get(
                "url"
            )

            if not url:
                continue

            self.audio(
                url=url,
                title=audio_data.get(
                    "title"
                ),
                mime_type=audio_data.get(
                    "mime_type",
                    audio_data.get(
                        "mimeType"
                    ),
                ),
            )

        # =================================================
        # FILES
        # =================================================

        for file_data in (
            context.get(
                "files",
                []
            )
            or []
        ):

            if not isinstance(
                file_data,
                dict
            ):
                continue

            name = file_data.get(
                "name"
            )

            if not name:
                continue

            self.file(
                name=name,
                url=file_data.get(
                    "url"
                ),
                path=file_data.get(
                    "path"
                ),
                mime_type=file_data.get(
                    "mime_type",
                    file_data.get(
                        "mimeType"
                    ),
                ),
                size=file_data.get(
                    "size"
                ),
                preview=file_data.get(
                    "preview"
                ),
            )

        # =================================================
        # ACTIONS
        # =================================================

        for action_data in (
            context.get(
                "actions",
                []
            )
            or []
        ):

            if not isinstance(
                action_data,
                dict
            ):
                continue

            name = action_data.get(
                "name"
            )

            if not name:
                continue

            self.action(
                name=name,
                status=action_data.get(
                    "status",
                    "success",
                ),
                message=action_data.get(
                    "message"
                ),
                data=action_data.get(
                    "data",
                    action_data.get(
                        "parameters"
                    ),
                ),
            )

        # =================================================
        # TOOL RESULTS
        # =================================================

        for tool_data in (
            context.get(
                "tool_results",
                []
            )
            or []
        ):

            if not isinstance(
                tool_data,
                dict
            ):
                continue

            tool = tool_data.get(
                "tool"
            )

            if not tool:
                continue

            self.tool_result(
                tool=tool,
                result=tool_data.get(
                    "result"
                ),
                success=tool_data.get(
                    "success",
                    True,
                ),
            )

        # =================================================
        # STATUS
        # =================================================

        for status_data in (
            context.get(
                "statuses",
                []
            )
            or []
        ):

            if not isinstance(
                status_data,
                dict
            ):
                continue

            status_text = status_data.get(
                "text"
            )

            if not status_text:
                continue

            self.status(
                text=status_text,
                state=status_data.get(
                    "state",
                    "working",
                ),
            )

        # =================================================
        # ERRORS
        # =================================================

        error_data = context.get(
            "error"
        )

        if error_data:

            if isinstance(
                error_data,
                list
            ):

                for item in error_data:

                    if isinstance(
                        item,
                        dict
                    ):

                        self.error(
                            message=item.get(
                                "message",
                                "An error occurred.",
                            ),
                            code=item.get(
                                "code"
                            ),
                            recoverable=item.get(
                                "recoverable",
                                True,
                            ),
                        )

                    else:

                        self.error(
                            str(item)
                        )

            elif isinstance(
                error_data,
                dict
            ):

                self.error(
                    message=error_data.get(
                        "message",
                        "An error occurred.",
                    ),
                    code=error_data.get(
                        "code"
                    ),
                    recoverable=error_data.get(
                        "recoverable",
                        True,
                    ),
                )

            else:

                self.error(
                    str(error_data)
                )

        return self

    # =====================================================
    # BUILD
    # =====================================================

    def build(
        self,
    ) -> Dict[str, Any]:

        metadata = dict(
            self._metadata
        )

        metadata.setdefault(
            "assistant",
            self.assistant_name,
        )

        metadata.setdefault(
            "version",
            "1.0",
        )

        return {
            "success": True,

            "content": [
                block.to_dict()
                for block in self._content
            ],

            "metadata": metadata,
        }

    def build_json(
        self,
    ) -> str:

        return json.dumps(
            self.build(),
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    # =====================================================
    # RESET
    # =====================================================

    def clear(
        self,
    ) -> "AnswerBuilder":

        self._content.clear()

        self._metadata.clear()

        return self


# =========================================================
# BUILD ANSWER
# =========================================================

def build_answer(
    model_response: Optional[str] = None,
    context: Optional[
        Dict[str, Any]
    ] = None,
    assistant_name: str = "IraAI",
) -> Dict[str, Any]:

    builder = AnswerBuilder(
        assistant_name=assistant_name
    )

    builder.set_metadata(
        "identity",
        "IraAI",
    )

    builder.set_metadata(
        "user_address",
        "Boss",
    )

    if context:

        builder.add_context(
            context
        )

    if model_response:

        builder.add_model_text(
            model_response
        )

    return builder.build()


# =========================================================
# ERROR ANSWER
# =========================================================

def build_error_answer(
    message: str,
    code: Optional[str] = None,
    assistant_name: str = "IraAI",
) -> Dict[str, Any]:

    builder = AnswerBuilder(
        assistant_name=assistant_name
    )

    builder.set_metadata(
        "identity",
        "IraAI",
    )

    builder.set_metadata(
        "user_address",
        "Boss",
    )

    builder.error(
        message=message,
        code=code,
    )

    return builder.build()


# =========================================================
# SUCCESS ANSWER
# =========================================================

def build_success_answer(
    text: str,
    assistant_name: str = "IraAI",
) -> Dict[str, Any]:

    builder = AnswerBuilder(
        assistant_name=assistant_name
    )

    builder.set_metadata(
        "identity",
        "IraAI",
    )

    builder.set_metadata(
        "user_address",
        "Boss",
    )

    builder.add_model_text(
        text
    )

    return builder.build()


# =========================================================
# END
# =========================================================