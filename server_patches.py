#!/usr/bin/env python3
"""Serve patch-based viewer: index + one-patch-at-a-time HE and feature PNGs."""

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

app = FastAPI(title="Patch viewer")
G: dict = {}

ALLOWED_LAYERS = {"he", "overlay_cells", "cell_mask", "vasculature", "immune"}


@app.get("/")
async def root():
    p = Path(__file__).parent / "viewer_patches.html"
    return FileResponse(p) if p.exists() else JSONResponse({"error": "viewer_patches.html not found"}, status_code=404)


@app.get("/index.json")
async def index_route():
    processed = G.get("processed")
    if not processed:
        raise HTTPException(500, "Server not configured")
    path = processed / "index.json"
    if not path.exists():
        raise HTTPException(404, "index.json not found")
    with open(path) as f:
        data = json.load(f)
    return JSONResponse(data)


@app.get("/patches/{layer}/{patch_id}.png")
async def patch_image(layer: str, patch_id: str):
    if layer not in ALLOWED_LAYERS:
        raise HTTPException(400, f"Invalid layer: {layer}")
    processed = G.get("processed")
    if not processed:
        raise HTTPException(500, "Server not configured")
    path = processed / layer / f"{patch_id}.png"
    if not path.is_file():
        raise HTTPException(404, f"Patch not found: {layer}/{patch_id}.png")
    return FileResponse(path, media_type="image/png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="processed", help="Path to processed/ (index.json + he/, etc.)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    G["processed"] = Path(args.processed).resolve()
    if not (G["processed"] / "index.json").exists():
        print(f"Warning: {G['processed'] / 'index.json'} not found. Run patchify.py first.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
