"""
Chatbot conversational endpoint.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request
from medismart.chatbot.assistant import PharmacyAssistant

chat_bp = Blueprint("chat_bp", __name__, url_prefix="/api/chat")
assistant = PharmacyAssistant()


@chat_bp.route("/message", methods=["POST"])
def chat_message():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"success": False, "error": "Message cannot be empty."}), 400

    response = assistant.answer_query(message)
    return jsonify({
        "success": True,
        "reply": response["reply"],
        "intent": response.get("intent", "general"),
    })
