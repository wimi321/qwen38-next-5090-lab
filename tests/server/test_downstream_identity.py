from __future__ import annotations

from freetoken.server.api_models import ModelCard
from freetoken.server.api_server import app


def test_public_api_uses_downstream_identity() -> None:
    assert app.title == "Qwen3.8 Next 5090 Lab API"
    assert ModelCard(id="test", root="/model").owned_by == "qwen38-next-5090-lab"
