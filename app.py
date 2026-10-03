# =========================================================
# IraAI — Main Backend API
# =========================================================
#
# Handles:
#     - HTTP API
#     - Authentication
#     - Chat
#     - Memory
#     - Tools
#     - Model health
#     - Backend health
#     - Configuration status
#
# Final answer construction:
#     answer_builder.py ONLY
#
# External AI APIs:
#     NONE
# =========================================================

from __future__ import annotations

from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import (
    API_KEY_REQUIRED,
    DEBUG,
    HOST,
    IRAAI_API_KEY,
    MAX_CONTENT_LENGTH,
    PORT,
    DEFAULT_MODEL,
    MODEL_ROLES,
    get_config,
    validate_config,
)

from chat import (
    chat,
    get_chat_controller,
)

from memory import (
    get_memory_manager,
    memory_stats,
)

from models import (
    get_all_models,
    model_health,
)

from tools import (
    available_tools,
    execute_tool as execute_registered_tool,
)


# =========================================================
# APP INFORMATION
# =========================================================

APP_NAME = "IraAI"
APP_VERSION = "1.0.0"


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# =========================================================
# CORS
# =========================================================

CORS(app)


# =========================================================
# JSON ERROR HANDLING
# =========================================================

@app.errorhandler(400)
def bad_request(error):

    return jsonify({
        "success": False,
        "error": "Bad request.",
    }), 400


@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "success": False,
        "error": "Endpoint not found.",
    }), 404


@app.errorhandler(413)
def request_too_large(error):

    return jsonify({
        "success": False,
        "error": "Request payload is too large.",
    }), 413


@app.errorhandler(500)
def internal_error(error):

    return jsonify({
        "success": False,
        "error": "Internal server error.",
    }), 500


# =========================================================
# API KEY
# =========================================================

def check_api_key() -> bool:

    if not API_KEY_REQUIRED:
        return True

    supplied_key = request.headers.get(
        "X-IraAI-Key",
        "",
    ).strip()

    return bool(
        supplied_key
        and IRAAI_API_KEY
        and supplied_key == IRAAI_API_KEY
    )


def authorization_error():

    return jsonify({
        "success": False,
        "error": "Unauthorized.",
    }), 401


# =========================================================
# REQUEST HELPERS
# =========================================================

def get_json_body() -> Dict[str, Any]:

    data = request.get_json(
        silent=True,
    )

    if not isinstance(data, dict):
        return {}

    return data


def clean_string(value: Any) -> str:

    if value is None:
        return ""

    return str(value).strip()


# =========================================================
# ROOT
# =========================================================

@app.route("/", methods=["GET"])
def index():

    return jsonify({
        "success": True,
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "local_models": True,
        "offline_inference": True,
        "external_ai_api": False,
        "message": "IraAI backend is running.",
    })


# =========================================================
# HEALTH
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    validation = validate_config()

    try:

        ai_status = model_health()

    except Exception as error:

        ai_status = {
            "available": False,
            "local": True,
            "offline": True,
            "external_api": False,
            "error": (
                f"{error.__class__.__name__}: "
                f"{error}"
            ),
        }

    return jsonify({
        "success": True,
        "server": True,
        "ai": ai_status,
        "config_errors": validation,
        "offline": True,
        "external_ai_api": False,
    })


# =========================================================
# CONFIG STATUS
# =========================================================

@app.route("/api/config", methods=["GET"])
def config_status():

    if not check_api_key():
        return authorization_error()

    return jsonify({
        "success": True,
        "config": get_config(),
    })


# =========================================================
# CHAT
# =========================================================

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():

    if not check_api_key():
        return authorization_error()

    data = get_json_body()

    message = clean_string(
        data.get("message")
    )

    if not message:

        return jsonify({
            "success": False,
            "error": "Message is required.",
        }), 400

    conversation_id = clean_string(
        data.get("conversation_id")
    )

    user_id = clean_string(
        data.get("user_id")
    )

    model = clean_string(
        data.get("model")
    )

    role = clean_string(
        data.get("role")
    )

    try:

        result = chat(
            message=message,
            conversation_id=(
                conversation_id or None
            ),
            user_id=(
                user_id or None
            ),
            model=(
                model or None
            ),
            role=(
                role or None
            ),
        )

        return jsonify({
            "success": True,
            "data": result,
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "error": (
                f"{error.__class__.__name__}: "
                f"{error}"
            ),
        }), 500


# =========================================================
# CHAT STATUS
# =========================================================

@app.route("/api/chat/status", methods=["GET"])
def chat_status():

    if not check_api_key():
        return authorization_error()

    controller = get_chat_controller()

    return jsonify({
        "success": True,
        "assistant": controller.assistant_name,
        "conversation_count": len(
            controller.conversations
        ),
    })


# =========================================================
# GET CONVERSATION
# =========================================================

@app.route(
    "/api/conversation/<conversation_id>",
    methods=["GET"],
)
def get_conversation(conversation_id: str):

    if not check_api_key():
        return authorization_error()

    controller = get_chat_controller()

    conversation = controller.get_conversation(
        conversation_id
    )

    if conversation is None:

        return jsonify({
            "success": False,
            "error": "Conversation not found.",
        }), 404

    return jsonify({
        "success": True,
        "conversation_id": conversation_id,
        "conversation": conversation,
    })


# =========================================================
# DELETE CONVERSATION
# =========================================================

@app.route(
    "/api/conversation/<conversation_id>",
    methods=["DELETE"],
)
def delete_conversation(conversation_id: str):

    if not check_api_key():
        return authorization_error()

    memory_manager = get_memory_manager()

    deleted = memory_manager.delete_conversation(
        conversation_id
    )

    controller = get_chat_controller()

    controller.clear_conversation(
        conversation_id
    )

    if not deleted:

        return jsonify({
            "success": False,
            "error": "Conversation not found.",
        }), 404

    return jsonify({
        "success": True,
        "conversation_id": conversation_id,
    })


# =========================================================
# MEMORY SEARCH
# =========================================================

@app.route(
    "/api/memory/search",
    methods=["POST"],
)
def memory_search():

    if not check_api_key():
        return authorization_error()

    data = get_json_body()

    query = clean_string(
        data.get("query")
    )

    if not query:

        return jsonify({
            "success": False,
            "error": "Search query is required.",
        }), 400

    limit = data.get("limit")

    try:

        results = get_memory_manager().search(
            query=query,
            limit=limit,
        )

        return jsonify({
            "success": True,
            "query": query,
            "results": results,
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "error": (
                f"{error.__class__.__name__}: "
                f"{error}"
            ),
        }), 500


# =========================================================
# MEMORY STATS
# =========================================================

@app.route(
    "/api/memory/stats",
    methods=["GET"],
)
def memory_statistics():

    if not check_api_key():
        return authorization_error()

    return jsonify({
        "success": True,
        "stats": memory_stats(),
    })


# =========================================================
# LONG-TERM MEMORY — GET
# =========================================================

@app.route(
    "/api/memory/long-term",
    methods=["GET"],
)
def get_long_term_memory():

    if not check_api_key():
        return authorization_error()

    manager = get_memory_manager()

    key = clean_string(
        request.args.get("key")
    )

    if key:

        result = manager.get_long_term(
            key
        )

    else:

        result = manager.get_long_term()

    return jsonify({
        "success": True,
        "memory": result,
    })


# =========================================================
# LONG-TERM MEMORY — SAVE
# =========================================================

@app.route(
    "/api/memory/long-term",
    methods=["POST"],
)
def save_long_term_memory():

    if not check_api_key():
        return authorization_error()

    data = get_json_body()

    key = clean_string(
        data.get("key")
    )

    if not key:

        return jsonify({
            "success": False,
            "error": "Memory key is required.",
        }), 400

    value = data.get("value")

    category = clean_string(
        data.get(
            "category",
            "general",
        )
    )

    importance = data.get(
        "importance",
        1.0,
    )

    try:

        importance = float(
            importance
        )

    except (
        TypeError,
        ValueError,
    ):

        importance = 1.0

    saved = get_memory_manager().save_long_term(
        key=key,
        value=value,
        category=category,
        importance=importance,
    )

    return jsonify({
        "success": bool(saved),
        "key": key,
    })


# =========================================================
# LONG-TERM MEMORY — DELETE
# =========================================================

@app.route(
    "/api/memory/long-term",
    methods=["DELETE"],
)
def delete_long_term_memory():

    if not check_api_key():
        return authorization_error()

    key = clean_string(
        request.args.get("key")
    )

    if not key:

        return jsonify({
            "success": False,
            "error": "Memory key is required.",
        }), 400

    deleted = get_memory_manager().delete_long_term(
        key
    )

    if not deleted:

        return jsonify({
            "success": False,
            "key": key,
            "error": "Memory key not found.",
        }), 404

    return jsonify({
        "success": True,
        "key": key,
    })


# =========================================================
# TOOLS — LIST
# =========================================================

@app.route(
    "/api/tools",
    methods=["GET"],
)
def tools_list():

    if not check_api_key():
        return authorization_error()

    return jsonify({
        "success": True,
        "tools": available_tools(),
    })


# =========================================================
# TOOLS — EXECUTE
# =========================================================

@app.route(
    "/api/tools/execute",
    methods=["POST"],
)
def execute_tool():

    if not check_api_key():
        return authorization_error()

    data = get_json_body()

    tool_name = clean_string(
        data.get("tool")
    )

    if not tool_name:

        return jsonify({
            "success": False,
            "error": "Tool name is required.",
        }), 400

    arguments = data.get(
        "arguments",
        {},
    )

    if not isinstance(arguments, dict):

        return jsonify({
            "success": False,
            "error": "Tool arguments must be an object.",
        }), 400

    try:

        result = execute_registered_tool(
            tool_name,
            **arguments,
        )

    except Exception as error:

        return jsonify({
            "success": False,
            "error": (
                f"{error.__class__.__name__}: "
                f"{error}"
            ),
        }), 500

    status_code = (
        200
        if result.get("success", False)
        else 400
    )

    return jsonify(result), status_code


# =========================================================
# MODELS — LIST
# =========================================================

@app.route(
    "/api/models",
    methods=["GET"],
)
def model_list():

    if not check_api_key():
        return authorization_error()

    models = get_all_models()

    return jsonify({
        "success": True,
        "models": models,
        "primary": DEFAULT_MODEL,
        "roles": MODEL_ROLES,
    })


# =========================================================
# MODELS — HEALTH
# =========================================================

@app.route(
    "/api/models/health",
    methods=["GET"],
)
def models_health():

    if not check_api_key():
        return authorization_error()

    return jsonify({
        "success": True,
        "health": model_health(),
    })


# =========================================================
# BACKEND STATUS
# =========================================================

@app.route(
    "/api/status",
    methods=["GET"],
)
def backend_status():

    if not check_api_key():
        return authorization_error()

    validation = validate_config()

    try:

        health_data = model_health()

    except Exception as error:

        health_data = {
            "available": False,
            "local": True,
            "offline": True,
            "external_api": False,
            "error": (
                f"{error.__class__.__name__}: "
                f"{error}"
            ),
        }

    return jsonify({
        "success": True,
        "name": APP_NAME,
        "version": APP_VERSION,
        "offline": True,
        "external_ai_api": False,
        "config_valid": not bool(validation),
        "config_errors": validation,
        "models": health_data,
    })


# =========================================================
# API FALLBACK
# =========================================================

@app.route(
    "/api/<path:unknown_path>",
    methods=[
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
    ],
)
def unknown_api(unknown_path: str):

    return jsonify({
        "success": False,
        "error": "API endpoint not found.",
        "path": f"/api/{unknown_path}",
    }), 404


# =========================================================
# APPLICATION FACTORY
# =========================================================

def create_app() -> Flask:
    return app


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    print(
        f"{APP_NAME} v{APP_VERSION}"
    )

    print(
        f"Server: http://{HOST}:{PORT}"
    )

    print(
        "Mode: LOCAL MODEL INFERENCE"
    )

    print(
        "External AI API: DISABLED"
    )

    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG,
    )