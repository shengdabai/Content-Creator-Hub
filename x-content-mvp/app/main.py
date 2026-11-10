import json
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app import db
from app.config import APP_NAME, CORS_ALLOWED_ORIGINS, UPLOAD_DIR
from app.services.content_pipeline import extract_insight, generate_drafts, score_drafts
from app.services.ingest import extract_web_content, image_to_data_url, normalize_text
from app.services.style_engine import build_style_profile


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    db._migrate_score_columns()
    db.get_or_create_default_profile()
    yield


app = FastAPI(title=APP_NAME, lifespan=lifespan)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1"] + [
        urlsplit(origin).hostname for origin in CORS_ALLOWED_ORIGINS if urlsplit(origin).hostname
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.middleware("http")
async def reject_cross_origin_writes(request: Request, call_next):
    # CORS alone does not prevent cross-site HTML form submissions to localhost.
    origin = request.headers.get("origin")
    if request.method not in {"GET", "HEAD", "OPTIONS"} and origin is not None:
        if origin not in CORS_ALLOWED_ORIGINS:
            return JSONResponse(status_code=403, content={"error": "Untrusted request origin."})
    return await call_next(request)


def _split_samples_from_text(text: str) -> List[str]:
    if not text.strip():
        return []
    chunks = []
    for block in text.split("\n"):
        piece = block.strip()
        if piece:
            chunks.append(piece)
    return chunks


def _run_generation_pipeline(source_row: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    image_data_url = None
    if source_row.get("image_path"):
        img_path = Path(source_row["image_path"])
        if img_path.exists():
            image_data_url = image_to_data_url(img_path)

    insight = extract_insight(
        source_type=source_row["source_type"],
        title=source_row.get("title", ""),
        normalized_text=source_row.get("normalized_text", ""),
        url=source_row.get("url", ""),
        image_data_url=image_data_url,
    )
    saved_insight = db.save_insight(source_row["id"], insight)

    drafts_raw = generate_drafts(saved_insight, profile)
    saved_drafts = db.save_drafts(
        source_id=source_row["id"],
        insight_id=saved_insight["id"],
        profile_id=profile["id"],
        drafts=drafts_raw,
    )

    draft_for_scoring = [
        {"post_type": d["post_type"], "tone": d["tone"], "content": d["content"]} for d in saved_drafts
    ]
    scored = score_drafts(draft_for_scoring, saved_insight, profile)
    score_payload = []
    for item in scored:
        idx = item["index"]
        if idx < len(saved_drafts):
            score_payload.append(
                {
                    "draft_id": saved_drafts[idx]["id"],
                    "style_match": item["style_match"],
                    "clarity": item["clarity"],
                    "attention": item["attention"],
                    "novelty": item["novelty"],
                    "risk": item["risk"],
                    "engagement": item["engagement"],
                    "overall": item["overall"],
                    "rationale": item["rationale"],
                }
            )
    db.save_scores(score_payload)

    return {
        "insight": saved_insight,
        "draft_count": len(saved_drafts),
    }


@app.get("/")
def home(request: Request):
    profile = db.get_or_create_default_profile()
    stats = db.dashboard_stats()
    drafts = db.list_drafts(limit=12)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "profile": profile,
            "stats": stats,
            "drafts": drafts,
            "error": request.query_params.get("error", ""),
            "message": request.query_params.get("message", ""),
            "active_nav": "create",
        },
    )


@app.post("/sources/create")
async def create_source(
    request: Request,
    url: str = Form(""),
    text_input: str = Form(""),
    image_file: Optional[UploadFile] = File(None),
):
    try:
        raw_text = text_input.strip()
        extracted_title = ""
        extracted_text = ""
        normalized_url = url.strip()
        image_path = ""

        if normalized_url:
            try:
                extracted = extract_web_content(normalized_url)
                extracted_title = extracted["title"]
                extracted_text = extracted["text"]
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail="URL extraction failed.") from exc

        if image_file and image_file.filename:
            original_name = Path(image_file.filename).name
            ext = Path(original_name).suffix.lower()
            if ext not in ALLOWED_IMAGE_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported image type. Allowed: "
                    + ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS)),
                )
            safe_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}_{original_name}"
            target = UPLOAD_DIR / safe_name
            data = await image_file.read()
            target.write_bytes(data)
            image_path = str(target)

        normalized_text = normalize_text("\n\n".join([raw_text, extracted_text]).strip())
        source_type = "text"
        if normalized_url and image_path:
            source_type = "url_text_image"
        elif normalized_url:
            source_type = "url_text"
        elif image_path and normalized_text:
            source_type = "text_image"
        elif image_path:
            source_type = "image"

        if not normalized_url and not normalized_text and not image_path:
            return RedirectResponse(url="/?error=Please provide URL, text, or image.", status_code=303)

        source = db.create_source_item(
            source_type=source_type,
            title=extracted_title or "Untitled Source",
            raw_input=raw_text,
            url=normalized_url,
            image_path=image_path,
            normalized_text=normalized_text,
        )
        profile = db.get_or_create_default_profile()
        _run_generation_pipeline(source, profile)
        return RedirectResponse(url=f"/sources/{source['id']}", status_code=303)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return RedirectResponse(url="/?error=Failed to generate drafts. Check server log.", status_code=303)


@app.get("/sources/{source_id}")
def source_detail(source_id: int, request: Request):
    source = db.get_source_item(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    insight = db.get_latest_insight_for_source(source_id)
    drafts = db.list_source_drafts(source_id)
    return templates.TemplateResponse(
        request=request,
        name="source_detail.html",
        context={
            "request": request,
            "source": source,
            "insight": insight,
            "drafts": drafts,
            "active_nav": "create",
        },
    )


@app.get("/style")
def style_page(request: Request):
    profile = db.get_or_create_default_profile()
    samples = db.list_style_samples(profile["id"], limit=120)
    return templates.TemplateResponse(
        request=request,
        name="style.html",
        context={
            "request": request,
            "profile": profile,
            "samples": samples,
            "message": request.query_params.get("message", ""),
            "active_nav": "style",
        },
    )


@app.post("/style/import")
async def style_import(
    samples_text: str = Form(""),
    samples_file: Optional[UploadFile] = File(None),
):
    profile = db.get_or_create_default_profile()
    parsed_samples = _split_samples_from_text(samples_text)

    if samples_file and samples_file.filename:
        content = (await samples_file.read()).decode("utf-8", errors="ignore")
        parsed_samples.extend(_split_samples_from_text(content))

    count = db.add_style_samples(profile["id"], parsed_samples)
    return RedirectResponse(url=f"/style?message=Imported {count} samples.", status_code=303)


@app.post("/style/rebuild")
def style_rebuild(manual_guide: str = Form("")):
    profile = db.get_or_create_default_profile()
    samples = [row["post_text"] for row in db.list_style_samples(profile["id"], limit=200)]
    built = build_style_profile(samples=samples, manual_guide=manual_guide)
    db.upsert_profile(built, manual_guide=manual_guide, profile_id=profile["id"])
    return RedirectResponse(url="/style?message=Style profile rebuilt.", status_code=303)


@app.get("/drafts")
def drafts_page(request: Request):
    drafts = db.list_drafts(limit=200)
    q = request.query_params.get("q", "").strip().lower()
    sort = request.query_params.get("sort", "newest").strip()
    if q:
        drafts = [d for d in drafts if q in d["content"].lower() or q in d["post_type"].lower()]
    if sort == "score":
        drafts.sort(key=lambda d: float(d.get("overall", 0)), reverse=True)
    return templates.TemplateResponse(
        request=request,
        name="drafts.html",
        context={"request": request, "drafts": drafts, "q": q, "sort": sort, "active_nav": "drafts"},
    )


class SourcePayload(BaseModel):
    url: str = ""
    text_input: str = ""


@app.post("/api/sources")
def api_create_source(payload: SourcePayload):
    url = payload.url.strip()
    text_input = payload.text_input.strip()
    title = ""
    web_text = ""
    if url:
        try:
            extracted = extract_web_content(url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail="URL extraction failed.") from exc
        title = extracted["title"]
        web_text = extracted["text"]

    normalized_text = normalize_text("\n\n".join([text_input, web_text]))
    if not url and not normalized_text:
        raise HTTPException(status_code=400, detail="Provide url or text_input.")

    source = db.create_source_item(
        source_type="url_text" if url else "text",
        title=title or "API Source",
        raw_input=text_input,
        url=url,
        image_path="",
        normalized_text=normalized_text,
    )
    profile = db.get_or_create_default_profile()
    pipeline_result = _run_generation_pipeline(source, profile)
    return JSONResponse({"source": source, "pipeline": pipeline_result})


@app.post("/api/extract/{source_id}")
def api_extract_only(source_id: int):
    source = db.get_source_item(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    insight = extract_insight(
        source_type=source["source_type"],
        title=source["title"],
        normalized_text=source["normalized_text"],
        url=source["url"],
        image_data_url=image_to_data_url(Path(source["image_path"])) if source["image_path"] else None,
    )
    saved = db.save_insight(source_id, insight)
    return JSONResponse(saved)


class StyleImportPayload(BaseModel):
    samples: List[str]
    manual_guide: str = ""


@app.post("/api/style/import")
def api_style_import(payload: StyleImportPayload):
    profile = db.get_or_create_default_profile()
    count = db.add_style_samples(profile["id"], payload.samples)
    samples = [row["post_text"] for row in db.list_style_samples(profile["id"], limit=200)]
    built = build_style_profile(samples, payload.manual_guide)
    updated = db.upsert_profile(built, payload.manual_guide, profile_id=profile["id"])
    return JSONResponse({"imported": count, "profile": updated})


@app.post("/api/drafts/generate/{source_id}")
def api_generate_drafts(source_id: int):
    source = db.get_source_item(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    profile = db.get_or_create_default_profile()
    result = _run_generation_pipeline(source, profile)
    return JSONResponse(result)


@app.post("/api/drafts/score/{source_id}")
def api_score_latest(source_id: int):
    drafts = db.list_source_drafts(source_id)
    insight = db.get_latest_insight_for_source(source_id)
    profile = db.get_or_create_default_profile()
    if not drafts or not insight:
        raise HTTPException(status_code=400, detail="Missing drafts or insight.")

    minimal = [{"post_type": d["post_type"], "tone": d["tone"], "content": d["content"]} for d in drafts]
    scores = score_drafts(minimal, insight, profile)
    db.save_scores(
        [
            {
                "draft_id": drafts[s["index"]]["id"],
                "style_match": s["style_match"],
                "clarity": s["clarity"],
                "attention": s["attention"],
                "novelty": s["novelty"],
                "risk": s["risk"],
                "engagement": s["engagement"],
                "overall": s["overall"],
                "rationale": s["rationale"],
            }
            for s in scores
        ]
    )
    return JSONResponse({"scored": len(scores)})


@app.get("/api/drafts")
def api_list_drafts():
    return JSONResponse({"drafts": db.list_drafts(limit=200)})


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=500, content={"error": "Internal server error."})
    return RedirectResponse(url="/?error=Unexpected server error. Check logs.", status_code=303)
