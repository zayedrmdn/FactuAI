from __future__ import annotations

import sys
from pathlib import Path

import anyio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app


async def _run() -> None:
    async with app.router.lifespan_context(app):
        print("lifespan-ok")


if __name__ == "__main__":
    anyio.run(_run)
