from typing import List, Any, Optional, Dict

from pymongo import MongoClient
from langchain_core.messages import AIMessage, HumanMessage, ChatMessage, SystemMessage
from datetime import datetime
import json

def _has_key(obj: Any, target: str) -> bool:
    """Suche rekursiv nach dem Schlüssel – egal wie tief verschachtelt."""
    if isinstance(obj, dict):
        if target in obj:
            return True
        return any(_has_key(v, target) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_key(item, target) for item in obj)
    return False


def extract_esco_obj(text: str, key: str = "escoSkills") -> Optional[Dict[str, Any]]:
    """
    Durchläuft den String, decodiert jedes gültige JSON-Objekt
    und liefert das erste, das den Schlüssel *key* enthält.
    Berücksichtigt korrekt verschachtelte Klammern und Strings.
    Gibt None zurück, wenn nichts gefunden wird.
    """
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        brace = text.find("{", i)
        if brace == -1:                     # kein weiteres „{“ → fertig
            break
        try:
            obj, end = decoder.raw_decode(text[brace:])
        except json.JSONDecodeError:
            i = brace + 1                   # hier war kein gültiges JSON
            continue
        if _has_key(obj, key):
            return obj
        i = brace + end                     # hinter das gefundene Objekt springen
    return None



def log_message(messages_obj: List[ChatMessage],
                assistant_name: str):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["ChatsNew"]
    collection = db[assistant_name]

    if isinstance(messages_obj, List):
        for message in messages_obj:
            try:
                if isinstance(message, HumanMessage):
                    doc = {
                        "role": message.type,
                        "content": message.content,
                        "timestamp": datetime.isoformat(datetime.now()),
                    }
                elif isinstance(message, AIMessage):
                    doc = {
                        "role": message.type,
                        "content": message.content,
                        "timestamp": datetime.isoformat(datetime.now()),
                        "response_metadata": message.response_metadata,
                        "usage_metadata": message.usage_metadata,
                    }
                elif isinstance(message, SystemMessage):
                    doc = {
                        "role": message.type,
                        "content": message.content,
                        "timestamp": datetime.isoformat(datetime.now()),
                    }
                else:
                    doc = {
                        "content": message.content,
                        "timestamp": datetime.isoformat(datetime.now())
                    }
                collection.insert_one(doc)
            except Exception as e:
                doc = message.to_json()
                collection.insert_one(doc)


