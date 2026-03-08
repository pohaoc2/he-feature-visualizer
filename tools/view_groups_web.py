#!/usr/bin/env python3
"""Local web viewer for H&E + multiplex group overlays."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from io import BytesIO

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from PIL import Image

from utils.group_visualizer import GroupVisualizerCore


def _to_png_bytes(rgb_image) -> bytes:
    buf = BytesIO()
    Image.fromarray(rgb_image).save(buf, format="PNG")
    return buf.getvalue()


def _make_index_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>H&E Group Viewer</title>
  <style>
    :root {
      --bg: #121212;
      --panel: #1d1d1d;
      --text: #f2f2f2;
      --muted: #b8b8b8;
      --accent: #5ea4ff;
    }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(circle at 20% 20%, #1f1f1f, #0f0f0f);
      color: var(--text);
    }
    .wrap {
      max-width: 1500px;
      margin: 0 auto;
      padding: 20px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: 1fr auto auto auto;
      gap: 12px;
      align-items: center;
      background: var(--panel);
      border: 1px solid #333;
      border-radius: 10px;
      padding: 12px;
    }
    .group-buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    button.group-btn {
      border: 1px solid #3b3b3b;
      background: #262626;
      color: var(--text);
      border-radius: 999px;
      padding: 8px 12px;
      cursor: pointer;
    }
    button.group-btn.active {
      border-color: #77b5ff;
      box-shadow: 0 0 0 1px #77b5ff inset;
      background: #1f2c40;
    }
    button.group-btn[disabled] {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .control {
      font-size: 14px;
      color: var(--muted);
      white-space: nowrap;
    }
    .zoom-hint {
      font-size: 13px;
      color: #a9d0ff;
      white-space: nowrap;
    }
    .viewer {
      margin-top: 14px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid #333;
      border-radius: 10px;
      overflow: hidden;
    }
    .panel h3 {
      margin: 0;
      padding: 10px 12px;
      font-size: 14px;
      border-bottom: 1px solid #2c2c2c;
      color: #d6e8ff;
    }
    .panel img,
    .panel canvas {
      display: block;
      width: 100%;
      height: auto;
      background: black;
      min-height: 220px;
    }
    .image-wrap {
      position: relative;
    }
    .zoom-box {
      overflow: auto;
      height: 72vh;
      background: black;
      cursor: grab;
    }
    .zoom-box.dragging {
      cursor: grabbing;
    }
    .image-wrap img {
      display: block;
      width: 100%;
      height: auto;
      background: black;
    }
    #group-img {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
      pointer-events: none;
    }
    .legend-overlay {
      position: absolute;
      right: 8px;
      top: 8px;
      background: rgba(16, 16, 16, 0.78);
      border: 1px solid #666;
      border-radius: 8px;
      padding: 8px;
      max-width: 45%;
      max-height: 65%;
      overflow: auto;
      font-size: 11px;
      color: #f2f2f2;
      line-height: 1.2;
    }
    .legend-row {
      display: flex;
      align-items: center;
      gap: 6px;
      margin: 2px 0;
    }
    .legend-swatch {
      display: inline-block;
      width: 12px;
      height: 12px;
      border: 1px solid #ddd;
      flex: 0 0 auto;
    }
    .meta {
      margin-top: 12px;
      font-size: 13px;
      color: var(--muted);
      background: var(--panel);
      border: 1px solid #333;
      border-radius: 10px;
      padding: 12px;
      line-height: 1.45;
    }
    .hint {
      color: #ffb76c;
    }
    @media (max-width: 960px) {
      .toolbar {
        grid-template-columns: 1fr;
      }
      .viewer {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="toolbar">
      <div id="group-buttons" class="group-buttons"></div>
      <div class="control">
        Combine
        <select id="combine-mode">
          <option value="max">max</option>
        </select>
      </div>
      <div class="control">
        Transparency
        <input id="alpha" type="range" min="0.05" max="1" step="0.01" value="1.00" />
        <span id="alpha-value">1.00</span>
      </div>
      <div id="zoom-readout" class="zoom-hint">Zoom: 0.75x (base) | Pinch/Ctrl+Wheel zoom, two-finger scroll pans</div>
    </div>

    <div class="viewer">
      <div class="panel">
        <h3>H&E</h3>
        <div id="he-zoom-box" class="zoom-box">
          <img id="he-img" alt="he" />
        </div>
      </div>
      <div class="panel">
        <h3 id="group-title">Multiplex Group</h3>
        <div id="group-zoom-box" class="zoom-box">
          <div id="group-scene" class="image-wrap">
            <img id="group-he-img" alt="he-underlay" />
            <img id="group-img" alt="group" />
            <div id="legend-overlay" class="legend-overlay">Loading legend...</div>
          </div>
        </div>
      </div>
    </div>

    <div id="meta" class="meta">Loading metadata...</div>
  </div>

  <script>
    const state = {
      groupId: "he",
      combineMode: "max",
      alpha: 1.0,
      zoom: 1.0,
      tier: "base",
      loadedTier: null,
      meta: null,
      groupMeta: null,
    };

    const heImg = document.getElementById("he-img");
    const heZoomBox = document.getElementById("he-zoom-box");
    const groupHeImg = document.getElementById("group-he-img");
    const groupImg = document.getElementById("group-img");
    const groupScene = document.getElementById("group-scene");
    const groupZoomBox = document.getElementById("group-zoom-box");
    const MIN_ZOOM = 0.35;
    const MAX_ZOOM = 8.0;
    let scrollSyncLock = false;
    let tierLoadToken = 0;

    function byId(id) {
      return document.getElementById(id);
    }

    function clamp(v, lo, hi) {
      return Math.max(lo, Math.min(hi, v));
    }

    function tierForZoom(zoom) {
      // Keep a stable render tier and let backend enforce level >= 1.
      return "base";
    }

    function captureScrollRatio(box) {
      const maxX = Math.max(0, box.scrollWidth - box.clientWidth);
      const maxY = Math.max(0, box.scrollHeight - box.clientHeight);
      return {
        x: maxX > 0 ? box.scrollLeft / maxX : 0,
        y: maxY > 0 ? box.scrollTop / maxY : 0,
      };
    }

    function applyScrollRatio(box, ratio) {
      const maxX = Math.max(0, box.scrollWidth - box.clientWidth);
      const maxY = Math.max(0, box.scrollHeight - box.clientHeight);
      box.scrollLeft = clamp((ratio?.x ?? 0) * maxX, 0, maxX);
      box.scrollTop = clamp((ratio?.y ?? 0) * maxY, 0, maxY);
    }

    function syncScrollFrom(source, target) {
      if (scrollSyncLock) {
        return;
      }
      scrollSyncLock = true;
      applyScrollRatio(target, captureScrollRatio(source));
      scrollSyncLock = false;
    }

    function setScrollToAnchor(box, anchorX, anchorY, clientX, clientY) {
      const maxX = Math.max(0, box.scrollWidth - box.clientWidth);
      const maxY = Math.max(0, box.scrollHeight - box.clientHeight);
      const desiredLeft = anchorX * Math.max(1, box.scrollWidth) - clientX;
      const desiredTop = anchorY * Math.max(1, box.scrollHeight) - clientY;
      box.scrollLeft = clamp(desiredLeft, 0, maxX);
      box.scrollTop = clamp(desiredTop, 0, maxY);
    }

    function applyAlpha() {
      groupImg.style.opacity = `${state.alpha}`;
      byId("alpha-value").textContent = Number(state.alpha).toFixed(2);
    }

    function updateZoomReadout() {
      byId("zoom-readout").textContent =
        `Zoom: ${state.zoom.toFixed(2)}x (${state.tier}) | Pinch/Ctrl+Wheel zoom, two-finger scroll pans`;
    }

    function setZoom(newZoom, options = null) {
      const z = clamp(Number(newZoom), MIN_ZOOM, MAX_ZOOM);
      const prevTier = state.tier;
      state.zoom = z;
      heImg.style.width = `${(z * 100).toFixed(1)}%`;
      groupScene.style.width = `${(z * 100).toFixed(1)}%`;
      state.tier = tierForZoom(z);
      updateZoomReadout();

      if (options && options.anchor) {
        const { sourceBox, sourceX, sourceY, anchorX, anchorY } = options.anchor;
        requestAnimationFrame(() => {
          setScrollToAnchor(sourceBox, anchorX, anchorY, sourceX, sourceY);
          const otherBox = sourceBox === heZoomBox ? groupZoomBox : heZoomBox;
          setScrollToAnchor(
            otherBox,
            anchorX,
            anchorY,
            otherBox.clientWidth / 2,
            otherBox.clientHeight / 2
          );
          syncScrollFrom(sourceBox, otherBox);
        });
      }

      if (prevTier !== state.tier) {
        const ratios = options?.anchor
          ? { x: options.anchor.anchorX, y: options.anchor.anchorY }
          : captureScrollRatio(heZoomBox);
        reloadTierImages({ force: true, scrollRatio: ratios });
      }
    }

    function renderLegend(markerColors) {
      const legend = byId("legend-overlay");
      if (!markerColors || !markerColors.length) {
        legend.innerHTML = "<b>Color Legend</b><br>No multiplex markers";
        return;
      }
      const rows = markerColors.map(([name, rgb]) => {
        return `<div class="legend-row">` +
          `<span class="legend-swatch" style="background:rgb(${rgb[0]},${rgb[1]},${rgb[2]});"></span>` +
          `<span>${name}</span>` +
          `</div>`;
      }).join("");
      legend.innerHTML = `<b>Color Legend</b><br>${rows}`;
    }

    function updateMetaText() {
      const metaDiv = byId("meta");
      if (!state.meta) {
        metaDiv.textContent = "No metadata loaded.";
        return;
      }
      const g = state.meta.groups.find(x => x.id === state.groupId);
      if (!g) {
        metaDiv.textContent = "Selected group missing from metadata.";
        return;
      }
      const resolved = state.groupMeta || g;
      const used = resolved.used_markers.length ? resolved.used_markers.join(", ") : "None";
      let colors = "None";
      if (resolved.marker_colors && resolved.marker_colors.length) {
        colors = resolved.marker_colors
          .map(([name, rgb]) => `${name}:rgb(${rgb[0]},${rgb[1]},${rgb[2]})`)
          .join(", ");
      }
      renderLegend(resolved.marker_colors || []);
      const heLevel = resolved.he_level ?? state.meta.he_level;
      const mxLevel = resolved.multiplex_level ?? state.meta.multiplex_level;
      const heShape = resolved.he_shape ?? state.meta.he_shape;
      const mxShape = resolved.multiplex_shape ?? state.meta.multiplex_shape;
      const registered = resolved.registered ?? state.meta.registered;
      const regJson = resolved.registration_index_json ?? state.meta.registration_index_json;
      const header = `tier=${state.tier}, he_level=${heLevel}, mx_level=${mxLevel}, ` +
        `he_shape=${heShape.join("x")}, mx_shape=${mxShape.join("x")}, registered=${registered}`;
      const regSrc = regJson ? `<br>registration source: ${regJson}` : "";
      const cmap = resolved.colormap || g.colormap || state.meta.default_colormap || "turbo";
      metaDiv.innerHTML =
        `${header}<br><br><b>${g.label}</b>` +
        `<br>Colormap: ${cmap}` +
        `<br>Used markers: ${used}` +
        `<br>Marker colors: ${colors}` +
        regSrc;
      if (!g.available) {
        metaDiv.innerHTML += `<br><span class="hint">This group is unavailable in current metadata.</span>`;
      }
    }

    function setActiveButton() {
      document.querySelectorAll("button.group-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.groupId === state.groupId);
      });
    }

    function reloadGroupImage(options = {}) {
      const ts = Date.now();
      const scrollRatio = options.scrollRatio || captureScrollRatio(heZoomBox);
      const url =
        `/api/group.png?group_id=${encodeURIComponent(state.groupId)}` +
        `&combine_mode=${encodeURIComponent(state.combineMode)}` +
        `&tier=${encodeURIComponent(state.tier)}&_=${ts}`;
      state.groupMeta = null;
      groupImg.onload = () => {
        requestAnimationFrame(() => {
          applyScrollRatio(groupZoomBox, scrollRatio);
          syncScrollFrom(groupZoomBox, heZoomBox);
        });
      };
      groupImg.src = url;
      byId("group-title").textContent = `Group: ${state.groupId} (${state.tier})`;
      fetch(
        `/api/group-meta?group_id=${encodeURIComponent(state.groupId)}` +
        `&combine_mode=${encodeURIComponent(state.combineMode)}` +
        `&tier=${encodeURIComponent(state.tier)}&_=${ts}`
      )
        .then((resp) => resp.ok ? resp.json() : null)
        .then((payload) => {
          if (payload) {
            state.groupMeta = payload;
          }
          updateMetaText();
        })
        .catch((err) => {
          console.warn("group meta failed", err);
          updateMetaText();
        });
      updateMetaText();
      setActiveButton();
    }

    function reloadTierImages(options = {}) {
      const force = options.force === true;
      if (!force && state.loadedTier === state.tier) {
        return;
      }
      const scrollRatio = options.scrollRatio || captureScrollRatio(heZoomBox);
      const ts = Date.now();
      const token = ++tierLoadToken;
      const heUrl = `/api/he.png?tier=${encodeURIComponent(state.tier)}&_=${ts}`;
      let pending = 2;
      const onDone = () => {
        pending -= 1;
        if (pending > 0) {
          return;
        }
        if (token !== tierLoadToken) {
          return;
        }
        requestAnimationFrame(() => {
          applyScrollRatio(heZoomBox, scrollRatio);
          applyScrollRatio(groupZoomBox, scrollRatio);
          syncScrollFrom(heZoomBox, groupZoomBox);
        });
      };
      heImg.onload = onDone;
      groupHeImg.onload = onDone;
      heImg.src = heUrl;
      groupHeImg.src = heUrl;
      state.loadedTier = state.tier;
      reloadGroupImage({ scrollRatio: scrollRatio });
    }

    async function loadMeta() {
      const res = await fetch("/api/meta");
      if (!res.ok) {
        throw new Error("Failed to load /api/meta");
      }
      state.meta = await res.json();

      const btnWrap = byId("group-buttons");
      btnWrap.innerHTML = "";
      for (const g of state.meta.groups) {
        const btn = document.createElement("button");
        btn.className = "group-btn";
        btn.dataset.groupId = g.id;
        btn.textContent = g.available ? g.label : `${g.label} (NA)`;
        btn.disabled = !g.available;
        btn.onclick = () => {
          state.groupId = g.id;
          reloadGroupImage();
        };
        btnWrap.appendChild(btn);
      }

      if (!state.meta.groups.some(g => g.id === state.groupId && g.available)) {
        const firstAvailable = state.meta.groups.find(g => g.available);
        state.groupId = firstAvailable ? firstAvailable.id : "he";
      }

      updateMetaText();
      setActiveButton();
    }

    async function boot() {
      byId("combine-mode").onchange = (ev) => {
        state.combineMode = ev.target.value;
        reloadGroupImage();
      };
      byId("alpha").oninput = (ev) => {
        state.alpha = parseFloat(ev.target.value);
        applyAlpha();
      };
      const wheelZoom = (ev, sourceBox) => {
        const wantsZoom = ev.ctrlKey || ev.metaKey || ev.altKey;
        if (!wantsZoom) {
          // Let native wheel/touchpad scrolling pan the FOV.
          return;
        }
        ev.preventDefault();
        const factor = ev.deltaY < 0 ? 1.14 : 1 / 1.14;
        const rect = sourceBox.getBoundingClientRect();
        const localX = clamp(ev.clientX - rect.left, 0, sourceBox.clientWidth);
        const localY = clamp(ev.clientY - rect.top, 0, sourceBox.clientHeight);
        const anchorX = (sourceBox.scrollLeft + localX) / Math.max(1, sourceBox.scrollWidth);
        const anchorY = (sourceBox.scrollTop + localY) / Math.max(1, sourceBox.scrollHeight);
        setZoom(state.zoom * factor, {
          anchor: {
            sourceBox: sourceBox,
            sourceX: localX,
            sourceY: localY,
            anchorX: clamp(anchorX, 0, 1),
            anchorY: clamp(anchorY, 0, 1),
          },
        });
      };
      const installDragPan = (box, peer) => {
        let dragging = false;
        let startX = 0;
        let startY = 0;
        let startLeft = 0;
        let startTop = 0;
        box.addEventListener("mousedown", (ev) => {
          if (ev.button !== 0) {
            return;
          }
          dragging = true;
          box.classList.add("dragging");
          startX = ev.clientX;
          startY = ev.clientY;
          startLeft = box.scrollLeft;
          startTop = box.scrollTop;
          ev.preventDefault();
        });
        window.addEventListener("mousemove", (ev) => {
          if (!dragging) {
            return;
          }
          box.scrollLeft = startLeft - (ev.clientX - startX);
          box.scrollTop = startTop - (ev.clientY - startY);
          syncScrollFrom(box, peer);
        });
        window.addEventListener("mouseup", () => {
          if (!dragging) {
            return;
          }
          dragging = false;
          box.classList.remove("dragging");
        });
      };
      heZoomBox.addEventListener("wheel", (ev) => wheelZoom(ev, heZoomBox), { passive: false });
      groupZoomBox.addEventListener("wheel", (ev) => wheelZoom(ev, groupZoomBox), {
        passive: false,
      });
      installDragPan(heZoomBox, groupZoomBox);
      installDragPan(groupZoomBox, heZoomBox);
      heZoomBox.addEventListener("scroll", () => syncScrollFrom(heZoomBox, groupZoomBox));
      groupZoomBox.addEventListener("scroll", () => syncScrollFrom(groupZoomBox, heZoomBox));
      applyAlpha();
      setZoom(0.75);
      updateZoomReadout();

      await loadMeta();
      reloadTierImages({ force: true, scrollRatio: { x: 0, y: 0 } });
    }

    boot().catch((err) => {
      byId("meta").textContent = `Startup error: ${err.message}`;
      console.error(err);
    });
  </script>
</body>
</html>
"""


def build_app(args: argparse.Namespace) -> FastAPI:
    effective_min_level = max(1, int(args.min_level))
    tier_max_dims: dict[str, int] = {
        "base": int(max(256, args.auto_max_dim)),
        "detail": int(max(256, args.auto_max_dim)),
        "high": int(max(256, args.auto_max_dim)),
    }

    cores: dict[str, GroupVisualizerCore] = {}
    core_summaries: dict[str, dict] = {}
    he_png_cache: dict[str, bytes] = {}
    group_png_cache: dict[tuple[str, str, str], bytes] = {}

    def _validate_tier(raw: str) -> str:
        tier = raw.strip().lower()
        if tier not in tier_max_dims:
            choices = ", ".join(tier_max_dims.keys())
            raise HTTPException(
                status_code=400,
                detail=f"invalid tier '{raw}'. Expected one of: {choices}",
            )
        return tier

    def _get_core(tier: str) -> GroupVisualizerCore:
        if tier in cores:
            return cores[tier]
        core = GroupVisualizerCore(
            he_image=args.he_image,
            multiplex_image=args.multiplex_image,
            metadata_csv=args.metadata_csv,
            he_level=args.he_level,
            multiplex_level=args.multiplex_level,
            auto_target_max_dim=tier_max_dims[tier],
            min_level=effective_min_level,
            default_colormap=args.default_colormap,
            index_json=args.index_json,
            preload_multiplex=args.preload_multiplex,
            contrast_gamma=args.contrast_gamma,
            contrast_gain=args.contrast_gain,
        )
        cores[tier] = core
        return core

    def _get_summary(tier: str) -> dict:
        if tier not in core_summaries:
            core_summaries[tier] = _get_core(tier).summary()
        return core_summaries[tier]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            for core in cores.values():
                core.close()

    app = FastAPI(title="H&E + Multiplex Group Viewer", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _make_index_html()

    @app.get("/api/meta")
    def api_meta() -> dict:
        summary = dict(_get_summary("base"))
        summary["available_tiers"] = list(tier_max_dims.keys())
        summary["tiers"] = {
            tier: {
                "auto_max_dim": tier_max_dims[tier],
                "loaded": tier in cores,
            }
            for tier in tier_max_dims
        }
        return summary

    @app.get("/api/he.png")
    def api_he_png(tier: str = Query("base")) -> Response:
        tier_key = _validate_tier(tier)
        if tier_key not in he_png_cache:
            he_png_cache[tier_key] = _to_png_bytes(_get_core(tier_key).he_rgb)
        return Response(content=he_png_cache[tier_key], media_type="image/png")

    @app.get("/api/group.png")
    def api_group_png(
        group_id: str = Query("immune"),
        combine_mode: str = Query("max"),
        tier: str = Query("base"),
    ) -> Response:
        tier_key = _validate_tier(tier)
        key = (
            tier_key,
            group_id.strip().lower(),
            combine_mode.strip().lower(),
        )
        if key in group_png_cache:
            return Response(content=group_png_cache[key], media_type="image/png")
        try:
            frame = _get_core(tier_key).render_group(
                group_id=group_id,
                combine_mode=combine_mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = _to_png_bytes(frame.image_rgb)
        group_png_cache[key] = payload
        return Response(content=payload, media_type="image/png")

    @app.get("/api/group-meta")
    def api_group_meta(
        group_id: str = Query("immune"),
        combine_mode: str = Query("max"),
        tier: str = Query("base"),
    ) -> dict:
        tier_key = _validate_tier(tier)
        core = _get_core(tier_key)
        summary = _get_summary(tier_key)
        try:
            frame = core.render_group(group_id=group_id, combine_mode=combine_mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "tier": tier_key,
            "group_id": frame.group_id,
            "label": frame.label,
            "used_markers": list(frame.used_markers),
            "missing_markers": list(frame.missing_markers),
            "colormap": frame.colormap,
            "he_level": int(summary["he_level"]),
            "multiplex_level": int(summary["multiplex_level"]),
            "he_shape": list(summary["he_shape"]),
            "multiplex_shape": list(summary["multiplex_shape"]),
            "registered": bool(summary["registered"]),
            "registration_index_json": summary["registration_index_json"],
            "marker_colors": [
                [name, [int(rgb[0]), int(rgb[1]), int(rgb[2])]]
                for name, rgb in frame.marker_colors
            ],
        }

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local web viewer for marker groups"
    )
    parser.add_argument(
        "--he-image",
        default="data/WD-76845-096.ome.tif",
        help="Path to H&E OME-TIFF",
    )
    parser.add_argument(
        "--multiplex-image",
        default="data/WD-76845-097.ome.tif",
        help="Path to multiplex OME-TIFF",
    )
    parser.add_argument(
        "--metadata-csv",
        default="data/WD-76845-097-metadata.csv",
        help="Path to metadata CSV",
    )
    parser.add_argument(
        "--index-json",
        default=None,
        help=(
            "Path to registration index.json containing warp_matrix. "
            "If omitted, auto-detects processed_wd/index.json or proceeded_wd/index.json."
        ),
    )
    parser.add_argument("--he-level", type=int, default=None, help="H&E pyramid level")
    parser.add_argument(
        "--multiplex-level", type=int, default=None, help="Multiplex pyramid level"
    )
    parser.add_argument(
        "--min-level",
        type=int,
        default=1,
        help="Minimum pyramid level for auto level selection",
    )
    parser.add_argument(
        "--auto-max-dim",
        type=int,
        default=1200,
        help="Target max side for auto level selection",
    )
    parser.add_argument(
        "--detail-auto-max-dim",
        type=int,
        default=3200,
        help="Target max side for detail tier auto level selection",
    )
    parser.add_argument(
        "--high-auto-max-dim",
        type=int,
        default=7000,
        help="Target max side for high tier auto level selection",
    )
    parser.add_argument(
        "--default-colormap",
        default="turbo",
        help="Fallback matplotlib colormap for group rendering",
    )
    parser.add_argument(
        "--contrast-gamma",
        type=float,
        default=0.65,
        help="Gamma (<1 brightens multiplex channels, default: 0.65)",
    )
    parser.add_argument(
        "--contrast-gain",
        type=float,
        default=1.35,
        help="Gain multiplier for multiplex brightness (default: 1.35)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Uvicorn host")
    parser.add_argument("--port", type=int, default=8010, help="Uvicorn port")
    parser.set_defaults(preload_multiplex=True)
    parser.add_argument(
        "--preload-multiplex",
        dest="preload_multiplex",
        action="store_true",
        help="Preload multiplex channels at selected level for faster switching (default: on)",
    )
    parser.add_argument(
        "--no-preload-multiplex",
        dest="preload_multiplex",
        action="store_false",
        help="Disable multiplex preloading to reduce memory",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
