"""
=========================================================
IraAI — Tools Manager
=========================================================

RESPONSIBILITY
--------------

This file manages tool integrations such as:

    - Web search
    - Web content analysis
    - File information
    - Image/video information
    - Android actions
    - Generic tool execution

IMPORTANT
---------

This file MUST NOT build the final user-facing answer.

It only returns structured tool results.

Final response construction belongs ONLY to:

    answer_builder.py
=========================================================
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse


# =========================================================
# TOOL RESULT
# =========================================================

def tool_success(
    tool: str,
    result: Any = None,
    **extra: Any
) -> Dict[str, Any]:

    response = {
        "success": True,
        "tool": str(tool),
        "result": result
    }

    response.update(extra)

    return response


def tool_error(
    tool: str,
    message: str,
    code: Optional[str] = None
) -> Dict[str, Any]:

    return {
        "success": False,

        "tool": str(tool),

        "error": {
            "message": str(message),

            "code": code
        }
    }


# =========================================================
# TOOLS MANAGER
# =========================================================

class ToolsManager:

    """
    Central tool registry.

    Tools are registered here and executed here.

    This class does NOT construct final AI answers.
    """

    def __init__(self):

        self._tools: Dict[
            str,
            Callable[..., Any]
        ] = {}

        self._register_builtin_tools()

    # =====================================================
    # REGISTER
    # =====================================================

    def register(
        self,
        name: str,
        function: Callable[..., Any]
    ) -> None:

        if not name:
            raise ValueError(
                "Tool name cannot be empty."
            )

        if not callable(function):
            raise TypeError(
                "Tool must be callable."
            )

        self._tools[
            str(name)
        ] = function

    # =====================================================
    # UNREGISTER
    # =====================================================

    def unregister(
        self,
        name: str
    ) -> bool:

        if name in self._tools:

            del self._tools[name]

            return True

        return False

    # =====================================================
    # LIST TOOLS
    # =====================================================

    def list_tools(
        self
    ) -> List[str]:

        return sorted(
            self._tools.keys()
        )

    # =====================================================
    # CHECK TOOL
    # =====================================================

    def has_tool(
        self,
        name: str
    ) -> bool:

        return name in self._tools

    # =====================================================
    # EXECUTE
    # =====================================================

    def execute(
        self,
        name: str,
        **kwargs: Any
    ) -> Dict[str, Any]:

        function = self._tools.get(
            name
        )

        if function is None:

            return tool_error(
                tool=name,

                message=(
                    f"Unknown tool: {name}"
                ),

                code="TOOL_NOT_FOUND"
            )

        try:

            result = function(
                **kwargs
            )

            if isinstance(
                result,
                dict
            ):

                if "success" in result:

                    return result

            return tool_success(
                tool=name,

                result=result
            )

        except Exception as error:

            return tool_error(
                tool=name,

                message=(
                    f"{error.__class__.__name__}: "
                    f"{error}"
                ),

                code="TOOL_EXECUTION_ERROR"
            )

    # =====================================================
    # PROCESS
    # =====================================================

    def process(
        self,
        request: str,
        tool_name: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:

        """
        Execute a specific tool when its name is known.

        Automatic tool selection should be handled by the
        orchestration/model layer rather than guessed here.
        """

        if not request:

            return tool_error(
                tool="tools",

                message="Tool request is empty.",

                code="EMPTY_REQUEST"
            )

        if not tool_name:

            return tool_success(
                tool="tools",

                result=None,

                message=(
                    "No specific tool was requested."
                )
            )

        return self.execute(
            tool_name,
            request=request,
            **kwargs
        )

    # =====================================================
    # BUILTIN TOOLS
    # =====================================================

    def _register_builtin_tools(
        self
    ) -> None:

        self.register(
            "web_search",
            self.web_search
        )

        self.register(
            "web_fetch",
            self.web_fetch
        )

        self.register(
            "file_info",
            self.file_info
        )

        self.register(
            "android_action",
            self.android_action
        )

    # =====================================================
    # WEB SEARCH
    # =====================================================

    def web_search(
        self,
        query: str,
        max_results: int = 5,
        **_: Any
    ) -> Dict[str, Any]:

        """
        Web-search adapter.

        The actual search provider can be connected later
        without changing chat.py or answer_builder.py.
        """

        query = str(
            query or ""
        ).strip()

        if not query:

            return tool_error(
                tool="web_search",

                message="Search query is empty.",

                code="EMPTY_QUERY"
            )

        return tool_success(
            tool="web_search",

            result=[],

            query=query,

            max_results=max(
                1,
                int(max_results)
            ),

            provider=None,

            status="not_configured"
        )

    # =====================================================
    # WEB FETCH
    # =====================================================

    def web_fetch(
        self,
        url: str,
        **_: Any
    ) -> Dict[str, Any]:

        """
        Web-page fetching adapter.

        A real HTTP/search provider should be connected
        here when the backend web tool is selected.
        """

        url = str(
            url or ""
        ).strip()

        if not url:

            return tool_error(
                tool="web_fetch",

                message="URL is empty.",

                code="EMPTY_URL"
            )

        parsed = urlparse(
            url
        )

        if parsed.scheme not in {
            "http",
            "https"
        }:

            return tool_error(
                tool="web_fetch",

                message="Only HTTP/HTTPS URLs are supported.",

                code="INVALID_URL"
            )

        return tool_success(
            tool="web_fetch",

            result=None,

            url=url,

            status="not_configured"
        )

    # =====================================================
    # FILE INFORMATION
    # =====================================================

    def file_info(
        self,
        path: str,
        **_: Any
    ) -> Dict[str, Any]:

        """
        Return safe metadata about a file.

        This does not read or modify the file contents.
        """

        if not path:

            return tool_error(
                tool="file_info",

                message="File path is empty.",

                code="EMPTY_PATH"
            )

        file_path = Path(
            str(path)
        )

        try:

            exists = file_path.exists()

            if not exists:

                return tool_error(
                    tool="file_info",

                    message=(
                        f"File does not exist: {path}"
                    ),

                    code="FILE_NOT_FOUND"
                )

            is_file = file_path.is_file()

            is_directory = file_path.is_dir()

            size = (
                file_path.stat().st_size
                if is_file
                else None
            )

            mime_type = (
                mimetypes.guess_type(
                    str(file_path)
                )[0]
                if is_file
                else None
            )

            return tool_success(
                tool="file_info",

                result={
                    "name":
                        file_path.name,

                    "path":
                        str(file_path),

                    "is_file":
                        is_file,

                    "is_directory":
                        is_directory,

                    "size":
                        size,

                    "mime_type":
                        mime_type
                }
            )

        except Exception as error:

            return tool_error(
                tool="file_info",

                message=(
                    f"{error.__class__.__name__}: "
                    f"{error}"
                ),

                code="FILE_INFO_ERROR"
            )

    # =====================================================
    # ANDROID ACTION
    # =====================================================

    def android_action(
        self,
        action: str,
        parameters: Optional[
            Dict[str, Any]
        ] = None,
        **_: Any
    ) -> Dict[str, Any]:

        """
        Android action adapter.

        The backend cannot directly control the Android
        device unless a secure Android bridge is connected.

        This method therefore returns an action request
        instead of falsely claiming that the action happened.
        """

        action = str(
            action or ""
        ).strip()

        if not action:

            return tool_error(
                tool="android_action",

                message="Android action is empty.",

                code="EMPTY_ACTION"
            )

        if parameters is None:

            parameters = {}

        return tool_success(
            tool="android_action",

            result={
                "action":
                    action,

                "parameters":
                    parameters
            },

            status="pending_android_execution"
        )

    # =====================================================
    # TOOL CONTEXT
    # =====================================================

    def build_context(
        self,
        results: List[
            Dict[str, Any]
        ]
    ) -> Dict[str, Any]:

        """
        Convert tool results into context that can later
        be passed to AnswerBuilder.
        """

        context: Dict[
            str,
            Any
        ] = {}

        sources = []

        images = []

        videos = []

        files = []

        actions = []

        errors = []

        for item in results:

            if not isinstance(
                item,
                dict
            ):
                continue

            if not item.get(
                "success",
                False
            ):

                error = item.get(
                    "error"
                )

                if error:

                    errors.append(
                        error
                    )

                continue

            result = item.get(
                "result"
            )

            tool_name = item.get(
                "tool"
            )

            # -----------------------------------------
            # WEB
            # -----------------------------------------

            if tool_name in {
                "web_search",
                "web_fetch"
            }:

                if isinstance(
                    result,
                    list
                ):

                    for source in result:

                        if isinstance(
                            source,
                            dict
                        ):

                            sources.append(
                                source
                            )

                elif isinstance(
                    result,
                    dict
                ):

                    source = result.get(
                        "source"
                    )

                    if isinstance(
                        source,
                        dict
                    ):

                        sources.append(
                            source
                        )

            # -----------------------------------------
            # FILE
            # -----------------------------------------

            if tool_name == "file_info":

                if isinstance(
                    result,
                    dict
                ):

                    files.append(
                        {
                            "name":
                                result.get(
                                    "name"
                                ),

                            "path":
                                result.get(
                                    "path"
                                ),

                            "mime_type":
                                result.get(
                                    "mime_type"
                                ),

                            "size":
                                result.get(
                                    "size"
                                )
                        }
                    )

            # -----------------------------------------
            # ANDROID
            # -----------------------------------------

            if tool_name == "android_action":

                actions.append(
                    result
                )

        if sources:

            context[
                "sources"
            ] = sources

        if images:

            context[
                "images"
            ] = images

        if videos:

            context[
                "videos"
            ] = videos

        if files:

            context[
                "files"
            ] = files

        if actions:

            context[
                "actions"
            ] = actions

        if errors:

            context[
                "error"
            ] = errors

        return context


# =========================================================
# GLOBAL TOOLS MANAGER
# =========================================================

_tools_manager: Optional[
    ToolsManager
] = None


def get_tools_manager() -> ToolsManager:

    global _tools_manager

    if _tools_manager is None:

        _tools_manager = ToolsManager()

    return _tools_manager


# =========================================================
# PUBLIC EXECUTION FUNCTION
# =========================================================

def execute_tool(
    name: str,
    **kwargs: Any
) -> Dict[str, Any]:

    return get_tools_manager().execute(
        name,
        **kwargs
    )


# =========================================================
# PUBLIC TOOL LIST
# =========================================================

def available_tools() -> List[str]:

    return get_tools_manager().list_tools()


# =========================================================
# END
# =========================================================