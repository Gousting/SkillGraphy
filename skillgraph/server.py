"""
Optional HTTP server for SkillGraph — multi-agent shared skill index.

Run:
    skillgraph serve --skills-dir ~/.hermes/skills --port 8100

API:
    POST /retrieve
    {"query": "generate a diagram", "top_k": 8}
    → [{"name": "excalidraw", "score": 0.92, ...}, ...]

    GET /info
    → {"total_skills": 147, "categories": {...}, ...}
"""

from __future__ import annotations

import json
from typing import Any


def create_app(retriever: Any):
    """Create a Flask-like WSGI app (uses httpx for zero deps).

    For production, use a proper WSGI server (gunicorn/uvicorn).
    """
    import httpx

    def app(environ: dict, start_response):
        path = environ.get("PATH_INFO", "/")
        method = environ.get("REQUEST_METHOD", "GET")

        if path == "/" and method == "GET":
            body = json.dumps({"status": "ok", "skills": len(retriever.entries)})
            start_response("200 OK", [("Content-Type", "application/json")])
            return [body.encode()]

        elif path == "/info" and method == "GET":
            body = json.dumps(retriever.stats, indent=2, ensure_ascii=False)
            start_response("200 OK", [("Content-Type", "application/json")])
            return [body.encode()]

        elif path == "/retrieve" and method == "POST":
            length = int(environ.get("CONTENT_LENGTH", 0))
            raw = environ["wsgi.input"].read(length).decode("utf-8")
            req = json.loads(raw)
            query = req.get("query", "")
            top_k = req.get("top_k", 8)

            results = retriever.retrieve(query, top_k=top_k)
            body = json.dumps(
                [
                    {
                        "name": s.name,
                        "description": s.description,
                        "category": s.category,
                        "score": round(s.score, 4),
                        "source": s.source,
                    }
                    for s in results
                ],
                ensure_ascii=False,
            )
            start_response("200 OK", [("Content-Type", "application/json")])
            return [body.encode()]

        else:
            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [b"Not Found"]

    return app


def run_server(
    retriever: Any,
    host: str = "0.0.0.0",
    port: int = 8100,
) -> None:
    """Run a simple HTTP server with the retriever."""
    from wsgiref.simple_server import make_server

    app = create_app(retriever)
    with make_server(host, port, app) as httpd:
        print(f"SkillGraph server running on http://{host}:{port}")
        print(f"  GET  /         — health check")
        print(f"  GET  /info     — index stats")
        print(f"  POST /retrieve — retrieve skills")
        httpd.serve_forever()