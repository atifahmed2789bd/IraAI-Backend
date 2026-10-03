# =========================================================
# IraAI — Memory Manager
# =========================================================
#
# RESPONSIBILITY
# --------------
#
# Conversation memory
# Long-term memory
# Memory search
# Facts
# Preferences
# Important context
# Memory save/update/delete
# Memory statistics
#
# IMPORTANT
# ---------
#
# This file does NOT:
#     - Generate AI answers
#     - Create system prompts
#     - Create personality instructions
#     - Route AI models
#
# Final user-facing answer construction belongs ONLY to:
#
#     answer_builder.py
#
# MEMORY POLICY
# -------------
#
#     - No fixed message limit
#     - No fixed conversation limit
#     - No automatic deletion
#     - search_limit=None means unlimited
#     - JSON storage
#     - Optional user_id support
# =========================================================

from __future__ import annotations

import json
import os
import threading
import time
import uuid

from pathlib import Path
from typing import Any, Dict, List, Optional


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MEMORY_FILE = Path(
    os.getenv(
        "IRAAI_MEMORY_FILE",
        str(BASE_DIR / "iraai_memory.json")
    )
)

MEMORY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOCK
# =========================================================

_memory_lock = threading.RLock()


# =========================================================
# TIME
# =========================================================

def current_timestamp() -> float:
    return time.time()


# =========================================================
# DEFAULT MEMORY
# =========================================================

def _create_default_memory() -> Dict[str, Any]:
    now = current_timestamp()

    return {
        "version": 1,
        "created_at": now,
        "updated_at": now,
        "conversations": [],
        "long_term_memory": {},
        "facts": [],
        "preferences": [],
        "important_context": [],
        "metadata": {}
    }


# =========================================================
# NORMALIZE MEMORY
# =========================================================

def _normalize_memory(memory: Any) -> Dict[str, Any]:

    if not isinstance(memory, dict):
        memory = {}

    now = current_timestamp()

    if not memory.get("version"):
        memory["version"] = 1

    if not memory.get("created_at"):
        memory["created_at"] = now

    if not memory.get("updated_at"):
        memory["updated_at"] = now

    if not isinstance(
        memory.get("conversations"),
        list
    ):
        memory["conversations"] = []

    if not isinstance(
        memory.get("long_term_memory"),
        dict
    ):
        memory["long_term_memory"] = {}

    if not isinstance(
        memory.get("facts"),
        list
    ):
        memory["facts"] = []

    if not isinstance(
        memory.get("preferences"),
        list
    ):
        memory["preferences"] = []

    if not isinstance(
        memory.get("important_context"),
        list
    ):
        memory["important_context"] = []

    if not isinstance(
        memory.get("metadata"),
        dict
    ):
        memory["metadata"] = {}

    return memory


# =========================================================
# INTERNAL READ
# =========================================================

def _read_memory_unlocked() -> Dict[str, Any]:

    if not MEMORY_FILE.exists():
        return _create_default_memory()

    try:

        with MEMORY_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return _normalize_memory(data)

    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError
    ):

        backup_file = MEMORY_FILE.with_name(
            MEMORY_FILE.stem
            + ".corrupt-"
            + str(int(current_timestamp()))
            + MEMORY_FILE.suffix
        )

        try:
            MEMORY_FILE.replace(backup_file)
        except OSError:
            pass

        return _create_default_memory()


# =========================================================
# INTERNAL WRITE
# =========================================================

def _write_memory_unlocked(
    memory: Dict[str, Any]
) -> bool:

    if not isinstance(memory, dict):
        return False

    memory = _normalize_memory(memory)

    memory["updated_at"] = current_timestamp()

    temp_file = MEMORY_FILE.with_name(
        MEMORY_FILE.name + ".tmp"
    )

    try:

        with temp_file.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                memory,
                file,
                ensure_ascii=False,
                indent=2
            )

            file.flush()
            os.fsync(file.fileno())

        temp_file.replace(MEMORY_FILE)

        return True

    except (
        OSError,
        TypeError,
        ValueError
    ):

        try:

            if temp_file.exists():
                temp_file.unlink()

        except OSError:
            pass

        return False


# =========================================================
# LOAD MEMORY
# =========================================================

def load_memory() -> Dict[str, Any]:

    with _memory_lock:

        memory = _read_memory_unlocked()

        if not MEMORY_FILE.exists():
            _write_memory_unlocked(memory)

        return memory


# =========================================================
# SAVE MEMORY
# =========================================================

def save_memory(
    memory: Dict[str, Any]
) -> bool:

    if not isinstance(memory, dict):
        return False

    with _memory_lock:
        return _write_memory_unlocked(memory)


# =========================================================
# FIND CONVERSATION
# =========================================================

def _find_conversation(
    memory: Dict[str, Any],
    conversation_id: str,
    user_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:

    conversations = memory.get(
        "conversations",
        []
    )

    if not isinstance(conversations, list):
        return None

    conversation_id = str(
        conversation_id or ""
    ).strip()

    normalized_user_id = (
        str(user_id).strip()
        if user_id is not None
        else None
    )

    for conversation in conversations:

        if not isinstance(conversation, dict):
            continue

        if conversation.get(
            "conversation_id"
        ) != conversation_id:
            continue

        if normalized_user_id is not None:

            stored_user_id = conversation.get(
                "user_id"
            )

            if (
                stored_user_id is not None
                and str(stored_user_id).strip()
                != normalized_user_id
            ):
                continue

        return conversation

    return None


# =========================================================
# SAVE CONVERSATION MESSAGE
# =========================================================

def save_conversation(
    conversation_id: Optional[str],
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:

    if conversation_id:
        conversation_id = str(
            conversation_id
        ).strip()
    else:
        conversation_id = str(uuid.uuid4())

    role = str(
        role or "user"
    ).strip()

    content = str(
        content or ""
    )

    if not role:
        role = "user"

    normalized_user_id = (
        str(user_id).strip()
        if user_id is not None
        else None
    )

    with _memory_lock:

        memory = _read_memory_unlocked()

        conversation = _find_conversation(
            memory,
            conversation_id,
            normalized_user_id
        )

        now = current_timestamp()

        if conversation is None:

            conversation = {
                "conversation_id": conversation_id,
                "created_at": now,
                "updated_at": now,
                "messages": [],
                "metadata": {}
            }

            if normalized_user_id:
                conversation["user_id"] = (
                    normalized_user_id
                )

            memory["conversations"].append(
                conversation
            )

        if not isinstance(
            conversation.get("messages"),
            list
        ):
            conversation["messages"] = []

        message = {
            "message_id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": now
        }

        if (
            isinstance(metadata, dict)
            and metadata
        ):
            message["metadata"] = dict(metadata)

        conversation["messages"].append(
            message
        )

        conversation["updated_at"] = now

        _write_memory_unlocked(memory)

        return message


# =========================================================
# GET CONVERSATION
# =========================================================

def get_conversation(
    conversation_id: str,
    user_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:

    conversation_id = str(
        conversation_id or ""
    ).strip()

    if not conversation_id:
        return None

    with _memory_lock:

        memory = _read_memory_unlocked()

        conversation = _find_conversation(
            memory,
            conversation_id,
            user_id
        )

        if conversation is None:
            return None

        return json.loads(
            json.dumps(
                conversation,
                ensure_ascii=False
            )
        )


# =========================================================
# GET CONVERSATION MESSAGES
# =========================================================

def get_conversation_messages(
    conversation_id: str,
    user_id: Optional[str] = None
) -> List[Dict[str, Any]]:

    conversation = get_conversation(
        conversation_id,
        user_id
    )

    if not conversation:
        return []

    messages = conversation.get(
        "messages",
        []
    )

    if not isinstance(messages, list):
        return []

    return messages


# =========================================================
# SAVE LONG-TERM MEMORY
# =========================================================

def save_long_term_memory(
    key: str,
    value: Any,
    category: str = "general",
    importance: float = 1.0,
    user_id: Optional[str] = None
) -> bool:

    key = str(
        key or ""
    ).strip()

    if not key:
        return False

    try:
        importance = float(importance)
    except (
        TypeError,
        ValueError
    ):
        importance = 1.0

    with _memory_lock:

        memory = _read_memory_unlocked()

        memory["long_term_memory"][key] = {
            "value": value,
            "category": str(
                category or "general"
            ),
            "importance": importance,
            "updated_at": current_timestamp()
        }

        if user_id is not None:
            memory["long_term_memory"][key][
                "user_id"
            ] = str(user_id).strip()

        return _write_memory_unlocked(memory)


# =========================================================
# GET LONG-TERM MEMORY
# =========================================================

def get_long_term_memory(
    key: Optional[str] = None,
    user_id: Optional[str] = None
) -> Any:

    with _memory_lock:

        memory = _read_memory_unlocked()

        long_term = memory.get(
            "long_term_memory",
            {}
        )

        if key is not None:

            item = long_term.get(
                str(key).strip()
            )

            if (
                item is not None
                and user_id is not None
                and isinstance(item, dict)
                and item.get("user_id") is not None
                and str(item.get("user_id"))
                != str(user_id)
            ):
                return None

            return item

        if user_id is None:
            return long_term

        return {
            memory_key: item
            for memory_key, item in long_term.items()
            if (
                isinstance(item, dict)
                and (
                    item.get("user_id") is None
                    or str(item.get("user_id"))
                    == str(user_id)
                )
            )
        }


# =========================================================
# DELETE LONG-TERM MEMORY
# =========================================================

def delete_long_term_memory(
    key: str,
    user_id: Optional[str] = None
) -> bool:

    key = str(
        key or ""
    ).strip()

    if not key:
        return False

    with _memory_lock:

        memory = _read_memory_unlocked()

        long_term = memory[
            "long_term_memory"
        ]

        if key not in long_term:
            return False

        item = long_term[key]

        if (
            user_id is not None
            and isinstance(item, dict)
            and item.get("user_id") is not None
            and str(item.get("user_id"))
            != str(user_id)
        ):
            return False

        del long_term[key]

        return _write_memory_unlocked(memory)


# =========================================================
# ADD FACT
# =========================================================

def add_fact(
    fact: str,
    source: str = "conversation",
    importance: float = 1.0,
    user_id: Optional[str] = None
) -> bool:

    fact = str(
        fact or ""
    ).strip()

    if not fact:
        return False

    try:
        importance = float(importance)
    except (
        TypeError,
        ValueError
    ):
        importance = 1.0

    item = {
        "id": str(uuid.uuid4()),
        "text": fact,
        "source": str(
            source or "conversation"
        ),
        "importance": importance,
        "created_at": current_timestamp()
    }

    if user_id is not None:
        item["user_id"] = str(user_id).strip()

    with _memory_lock:

        memory = _read_memory_unlocked()

        memory["facts"].append(item)

        return _write_memory_unlocked(memory)


# =========================================================
# ADD PREFERENCE
# =========================================================

def add_preference(
    preference: str,
    category: str = "general",
    user_id: Optional[str] = None
) -> bool:

    preference = str(
        preference or ""
    ).strip()

    if not preference:
        return False

    item = {
        "id": str(uuid.uuid4()),
        "text": preference,
        "category": str(
            category or "general"
        ),
        "created_at": current_timestamp()
    }

    if user_id is not None:
        item["user_id"] = str(user_id).strip()

    with _memory_lock:

        memory = _read_memory_unlocked()

        memory["preferences"].append(item)

        return _write_memory_unlocked(memory)


# =========================================================
# ADD IMPORTANT CONTEXT
# =========================================================

def add_important_context(
    context: str,
    source: str = "conversation",
    user_id: Optional[str] = None
) -> bool:

    context = str(
        context or ""
    ).strip()

    if not context:
        return False

    item = {
        "id": str(uuid.uuid4()),
        "text": context,
        "source": str(
            source or "conversation"
        ),
        "created_at": current_timestamp()
    }

    if user_id is not None:
        item["user_id"] = str(user_id).strip()

    with _memory_lock:

        memory = _read_memory_unlocked()

        memory["important_context"].append(item)

        return _write_memory_unlocked(memory)


# =========================================================
# MEMORY SEARCH
# =========================================================

def search_memory(
    query: str,
    limit: Optional[int] = None,
    user_id: Optional[str] = None
) -> List[Dict[str, Any]]:

    query = str(
        query or ""
    ).strip().lower()

    if not query:
        return []

    normalized_user_id = (
        str(user_id).strip()
        if user_id is not None
        else None
    )

    # -----------------------------------------------------
    # Read once.
    # No nested lock call.
    # -----------------------------------------------------

    with _memory_lock:
        memory = _read_memory_unlocked()

    results: List[Dict[str, Any]] = []

    # =====================================================
    # LONG-TERM MEMORY
    # =====================================================

    long_term = memory.get(
        "long_term_memory",
        {}
    )

    if isinstance(long_term, dict):

        for key, item in long_term.items():

            if not isinstance(item, dict):
                continue

            if (
                normalized_user_id is not None
                and item.get("user_id") is not None
                and str(item.get("user_id"))
                != normalized_user_id
            ):
                continue

            searchable = " ".join(
                [
                    str(key),
                    str(item.get("value", "")),
                    str(item.get("category", ""))
                ]
            ).lower()

            if query in searchable:

                results.append({
                    "type": "long_term_memory",
                    "key": key,
                    "value": item.get("value"),
                    "category": item.get("category"),
                    "importance": item.get(
                        "importance",
                        1.0
                    ),
                    "updated_at": item.get(
                        "updated_at"
                    ),
                    "user_id": item.get(
                        "user_id"
                    )
                })

    # =====================================================
    # FACTS
    # =====================================================

    for fact in memory.get("facts", []):

        if not isinstance(fact, dict):
            continue

        if (
            normalized_user_id is not None
            and fact.get("user_id") is not None
            and str(fact.get("user_id"))
            != normalized_user_id
        ):
            continue

        text = str(
            fact.get("text", "")
        )

        if query in text.lower():

            result = dict(fact)
            result["type"] = "fact"

            results.append(result)

    # =====================================================
    # PREFERENCES
    # =====================================================

    for preference in memory.get(
        "preferences",
        []
    ):

        if not isinstance(preference, dict):
            continue

        if (
            normalized_user_id is not None
            and preference.get("user_id") is not None
            and str(preference.get("user_id"))
            != normalized_user_id
        ):
            continue

        text = str(
            preference.get("text", "")
        )

        if query in text.lower():

            result = dict(preference)
            result["type"] = "preference"

            results.append(result)

    # =====================================================
    # IMPORTANT CONTEXT
    # =====================================================

    for item in memory.get(
        "important_context",
        []
    ):

        if not isinstance(item, dict):
            continue

        if (
            normalized_user_id is not None
            and item.get("user_id") is not None
            and str(item.get("user_id"))
            != normalized_user_id
        ):
            continue

        text = str(
            item.get("text", "")
        )

        if query in text.lower():

            result = dict(item)
            result["type"] = "important_context"

            results.append(result)

    # =====================================================
    # CONVERSATIONS
    # =====================================================

    for conversation in memory.get(
        "conversations",
        []
    ):

        if not isinstance(conversation, dict):
            continue

        stored_user_id = conversation.get(
            "user_id"
        )

        if (
            normalized_user_id is not None
            and stored_user_id is not None
            and str(stored_user_id)
            != normalized_user_id
        ):
            continue

        conversation_id = conversation.get(
            "conversation_id"
        )

        messages = conversation.get(
            "messages",
            []
        )

        if not isinstance(messages, list):
            continue

        for message in messages:

            if not isinstance(message, dict):
                continue

            content = str(
                message.get("content", "")
            )

            if query in content.lower():

                results.append({
                    "type": "conversation",
                    "conversation_id": conversation_id,
                    "user_id": stored_user_id,
                    "message_id": message.get(
                        "message_id"
                    ),
                    "role": message.get(
                        "role"
                    ),
                    "content": content,
                    "timestamp": message.get(
                        "timestamp"
                    )
                })

    # =====================================================
    # LIMIT
    # =====================================================

    if limit is None:
        return results

    try:
        limit = int(limit)
    except (
        TypeError,
        ValueError
    ):
        return results

    if limit <= 0:
        return results

    return results[:limit]


# =========================================================
# BUILD MEMORY CONTEXT
# =========================================================

def build_memory_context(
    query: str,
    conversation_id: Optional[str] = None,
    search_limit: Optional[int] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:

    # -----------------------------------------------------
    # Read memory once.
    # search_memory() is intentionally NOT called while
    # holding this lock.
    # -----------------------------------------------------

    with _memory_lock:
        memory = _read_memory_unlocked()

    context: Dict[str, Any] = {}

    normalized_user_id = (
        str(user_id).strip()
        if user_id is not None
        else None
    )

    # =====================================================
    # LONG-TERM MEMORY
    # =====================================================

    long_term = memory.get(
        "long_term_memory",
        {}
    )

    if isinstance(long_term, dict):

        if normalized_user_id is None:

            if long_term:
                context[
                    "long_term_memory"
                ] = long_term

        else:

            user_long_term = {
                key: item
                for key, item in long_term.items()
                if (
                    isinstance(item, dict)
                    and (
                        item.get("user_id") is None
                        or str(item.get("user_id"))
                        == normalized_user_id
                    )
                )
            }

            if user_long_term:
                context[
                    "long_term_memory"
                ] = user_long_term

    # =====================================================
    # SEARCH MATCHES
    # =====================================================

    if query:

        matches = search_memory(
            query=query,
            limit=search_limit,
            user_id=normalized_user_id
        )

        if matches:

            context[
                "memory_matches"
            ] = matches

    # =====================================================
    # CURRENT CONVERSATION
    # =====================================================

    if conversation_id:

        conversation = _find_conversation(
            memory,
            conversation_id,
            normalized_user_id
        )

        if conversation:

            context[
                "current_conversation"
            ] = conversation

    # =====================================================
    # FACTS
    # =====================================================

    facts = memory.get(
        "facts",
        []
    )

    if normalized_user_id is not None:

        facts = [
            item
            for item in facts
            if (
                isinstance(item, dict)
                and (
                    item.get("user_id") is None
                    or str(item.get("user_id"))
                    == normalized_user_id
                )
            )
        ]

    if facts:
        context["facts"] = facts

    # =====================================================
    # PREFERENCES
    # =====================================================

    preferences = memory.get(
        "preferences",
        []
    )

    if normalized_user_id is not None:

        preferences = [
            item
            for item in preferences
            if (
                isinstance(item, dict)
                and (
                    item.get("user_id") is None
                    or str(item.get("user_id"))
                    == normalized_user_id
                )
            )
        ]

    if preferences:
        context["preferences"] = preferences

    # =====================================================
    # IMPORTANT CONTEXT
    # =====================================================

    important_context = memory.get(
        "important_context",
        []
    )

    if normalized_user_id is not None:

        important_context = [
            item
            for item in important_context
            if (
                isinstance(item, dict)
                and (
                    item.get("user_id") is None
                    or str(item.get("user_id"))
                    == normalized_user_id
                )
            )
        ]

    if important_context:
        context[
            "important_context"
        ] = important_context

    return context


# =========================================================
# DELETE CONVERSATION
# =========================================================

def delete_conversation(
    conversation_id: str,
    user_id: Optional[str] = None
) -> bool:

    conversation_id = str(
        conversation_id or ""
    ).strip()

    if not conversation_id:
        return False

    with _memory_lock:

        memory = _read_memory_unlocked()

        conversations = memory.get(
            "conversations",
            []
        )

        if not isinstance(
            conversations,
            list
        ):
            return False

        original_count = len(
            conversations
        )

        normalized_user_id = (
            str(user_id).strip()
            if user_id is not None
            else None
        )

        memory["conversations"] = [
            item
            for item in conversations
            if not (
                isinstance(item, dict)
                and item.get(
                    "conversation_id"
                ) == conversation_id
                and (
                    normalized_user_id is None
                    or item.get("user_id") is None
                    or str(item.get("user_id"))
                    == normalized_user_id
                )
            )
        ]

        if len(
            memory["conversations"]
        ) == original_count:
            return False

        return _write_memory_unlocked(
            memory
        )


# =========================================================
# CLEAR ALL MEMORY
# =========================================================

def clear_all_memory(
    confirm: bool = False
) -> bool:

    if confirm is not True:
        return False

    with _memory_lock:

        memory = _create_default_memory()

        return _write_memory_unlocked(
            memory
        )


# =========================================================
# MEMORY STATISTICS
# =========================================================

def memory_stats() -> Dict[str, Any]:

    with _memory_lock:

        memory = _read_memory_unlocked()

        conversations = memory.get(
            "conversations",
            []
        )

        if not isinstance(
            conversations,
            list
        ):
            conversations = []

        message_count = 0

        for conversation in conversations:

            if not isinstance(
                conversation,
                dict
            ):
                continue

            messages = conversation.get(
                "messages",
                []
            )

            if isinstance(messages, list):
                message_count += len(messages)

        long_term = memory.get(
            "long_term_memory",
            {}
        )

        facts = memory.get(
            "facts",
            []
        )

        preferences = memory.get(
            "preferences",
            []
        )

        important_context = memory.get(
            "important_context",
            []
        )

        return {
            "memory_file": str(MEMORY_FILE),

            "conversations": len(
                conversations
            ),

            "messages": message_count,

            "long_term_memory": len(
                long_term
                if isinstance(
                    long_term,
                    dict
                )
                else {}
            ),

            "facts": len(
                facts
                if isinstance(
                    facts,
                    list
                )
                else []
            ),

            "preferences": len(
                preferences
                if isinstance(
                    preferences,
                    list
                )
                else []
            ),

            "important_context": len(
                important_context
                if isinstance(
                    important_context,
                    list
                )
                else []
            ),

            "storage": "json",

            "fixed_entry_limit": False,

            "automatic_deletion": False
        }


# =========================================================
# MEMORY MANAGER CLASS
# =========================================================

class MemoryManager:

    def load(
        self
    ) -> Dict[str, Any]:

        return load_memory()

    def save(
        self,
        memory: Dict[str, Any]
    ) -> bool:

        return save_memory(memory)

    def save_conversation(
        self,
        conversation_id: Optional[str],
        role: str,
        content: str,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:

        return save_conversation(
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata=metadata,
            user_id=user_id
        )

    def get_conversation(
        self,
        conversation_id: str,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:

        return get_conversation(
            conversation_id,
            user_id
        )

    def get_conversation_messages(
        self,
        conversation_id: str,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        return get_conversation_messages(
            conversation_id,
            user_id
        )

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        return search_memory(
            query=query,
            limit=limit,
            user_id=user_id
        )

    def build_context(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        search_limit: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:

        return build_memory_context(
            query=query,
            conversation_id=conversation_id,
            search_limit=search_limit,
            user_id=user_id
        )

    def save_long_term(
        self,
        key: str,
        value: Any,
        category: str = "general",
        importance: float = 1.0,
        user_id: Optional[str] = None
    ) -> bool:

        return save_long_term_memory(
            key=key,
            value=value,
            category=category,
            importance=importance,
            user_id=user_id
        )

    def get_long_term(
        self,
        key: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Any:

        return get_long_term_memory(
            key,
            user_id
        )

    def delete_long_term(
        self,
        key: str,
        user_id: Optional[str] = None
    ) -> bool:

        return delete_long_term_memory(
            key,
            user_id
        )

    def add_fact(
        self,
        fact: str,
        source: str = "conversation",
        importance: float = 1.0,
        user_id: Optional[str] = None
    ) -> bool:

        return add_fact(
            fact=fact,
            source=source,
            importance=importance,
            user_id=user_id
        )

    def add_preference(
        self,
        preference: str,
        category: str = "general",
        user_id: Optional[str] = None
    ) -> bool:

        return add_preference(
            preference=preference,
            category=category,
            user_id=user_id
        )

    def add_important_context(
        self,
        context: str,
        source: str = "conversation",
        user_id: Optional[str] = None
    ) -> bool:

        return add_important_context(
            context=context,
            source=source,
            user_id=user_id
        )

    def delete_conversation(
        self,
        conversation_id: str,
        user_id: Optional[str] = None
    ) -> bool:

        return delete_conversation(
            conversation_id,
            user_id
        )

    def clear_all(
        self,
        confirm: bool = False
    ) -> bool:

        return clear_all_memory(confirm)

    def stats(
        self
    ) -> Dict[str, Any]:

        return memory_stats()


# =========================================================
# GLOBAL MEMORY MANAGER
# =========================================================

_memory_manager: Optional[
    MemoryManager
] = None

_memory_manager_lock = threading.RLock()


def get_memory_manager() -> MemoryManager:

    global _memory_manager

    with _memory_manager_lock:

        if _memory_manager is None:

            _memory_manager = MemoryManager()

        return _memory_manager


# =========================================================
# END
# =========================================================