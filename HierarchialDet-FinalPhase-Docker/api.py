"""
FastAPI service for HierarchialDet — Dental X-ray abnormal-tooth detection.

Endpoints
---------
GET  /health   — liveness probe
GET  /version  — model and API metadata
POST /predict  — run inference on a single .mha file (multipart upload)

The model is loaded lazily on the first /predict call so the container
starts fast and /health is always reachable even before weights are ready.

Environment variables (all optional, defaults match the Docker layout)
----------------------------------------------------------------------
CONFIG_PATH       path to the diffdet YAML config
MODEL_WEIGHTS     path to model_final.pth
IMAGE_INDEX_PATH  path to test_ids.json
CONFIDENCE_THRESHOLD  float, detection score cutoff (default 0.0)
HOST              bind address (default 0.0.0.0)
PORT              bind port (default 8000)
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("hierarchialdet.api")

_API_VERSION = "1.0.0"
_MODEL_NAME = "HierarchialDet-FinalPhase"

app = FastAPI(
    title=_MODEL_NAME,
    description="Hierarchical diffusion-based detection of abnormal teeth on panoramic dental X-rays.",
    version=_API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

_demo: Optional[object] = None
_image_index: Optional[dict] = None


def _load_model():
    """Lazy-initialise the model. Thread-safe for single-worker deployments."""
    global _demo, _image_index

    if _demo is not None:
        return

    import SimpleITK  # noqa — verify import before heavy detectron2 load

    from detectron2.config import get_cfg
    from hierarchialdet import add_diffusiondet_config
    from hierarchialdet.util.model_ema import add_model_ema_configs
    from hierarchialdet.predictor import VisualizationDemo
    from process import load_image_index

    config_path = os.environ.get(
        "CONFIG_PATH",
        "/opt/app/configs/diffdet.custom.swinbase.nonpretrain.yaml",
    )
    weights_path = os.environ.get(
        "MODEL_WEIGHTS",
        "/opt/app/pretrained_model/model_final.pth",
    )
    confidence = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.0"))

    for path, label in [(config_path, "CONFIG_PATH"), (weights_path, "MODEL_WEIGHTS")]:
        if not os.path.isfile(path):
            raise RuntimeError(
                f"{label} not found: {path}. "
                "Set the environment variable to the correct path."
            )

    logger.info("Loading model configuration: %s", config_path)
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence
    cfg.freeze()

    logger.info("Loading model weights: %s", weights_path)
    _demo = VisualizationDemo(cfg, k=2)

    index_path = os.environ.get(
        "IMAGE_INDEX_PATH",
        str(Path(__file__).parent / "test_ids.json"),
    )
    try:
        _image_index = load_image_index(index_path)
        logger.info("Image index loaded: %d entries", len(_image_index))
    except FileNotFoundError as exc:
        logger.warning("Image index not found, IDs will default to slice index. %s", exc)
        _image_index = {}

    logger.info("Model ready.")


@app.get("/health", summary="Liveness probe")
def health():
    """Returns 200 OK as long as the server is running."""
    return {"status": "ok"}


@app.get("/version", summary="Model and API version metadata")
def version():
    """Returns API version and model metadata."""
    return {
        "api_version": _API_VERSION,
        "model": _MODEL_NAME,
        "model_loaded": _demo is not None,
        "config_path": os.environ.get("CONFIG_PATH", "/opt/app/configs/diffdet.custom.swinbase.nonpretrain.yaml"),
        "weights_path": os.environ.get("MODEL_WEIGHTS", "/opt/app/pretrained_model/model_final.pth"),
    }


@app.post(
    "/predict",
    summary="Run inference on a panoramic dental X-ray",
    response_description="Grand-challenge compatible detection JSON",
)
async def predict(file: UploadFile = File(..., description="Panoramic X-ray in .mha format")):
    """
    Accept a single **.mha** panoramic X-ray, run HierarchialDet inference,
    and return detections in Grand-Challenge bounding-box JSON format.

    Each detected abnormal tooth is annotated with:
    - `name`: `"<quadrant> - <enumeration> - <diagnosis>"`
    - `corners`: four 3-D corner points `[x, y, slice_id]`
    - `probability`: detection confidence score
    """
    if not file.filename.endswith(".mha"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only .mha files are accepted.",
        )

    try:
        _load_model()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {exc}",
        )

    import SimpleITK as sitk
    from process import custom_format_output

    with tempfile.NamedTemporaryFile(suffix=".mha", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    try:
        logger.info("Reading uploaded file: %s (%s)", file.filename, tmp_path)
        try:
            image = sitk.ReadImage(tmp_path)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to read .mha file: {exc}",
            )

        image_array = sitk.GetArrayFromImage(image)
        total_slices = image_array.shape[2]
        logger.info("Running inference on %d slice(s)", total_slices)

        all_outputs = []
        img_ids = []

        for k in range(total_slices):
            image_name = f"test_{k}.png"
            try:
                predictions, _ = _demo.run_on_image(image_array[:, :, k, :])
            except Exception as exc:
                logger.error("Inference failed on slice %d: %s", k, exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Inference failed on slice {k}: {exc}",
                )
            all_outputs.append(predictions["instances"])
            img_id = _image_index.get(image_name, k)
            img_ids.append(img_id)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    result = custom_format_output(all_outputs, img_ids)
    logger.info(
        "Inference complete: %d detection(s) across %d slice(s)",
        len(result.get("boxes", [])),
        total_slices,
    )
    return JSONResponse(content=result)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    logger.info("Starting HierarchialDet API on %s:%d", host, port)
    uvicorn.run("api:app", host=host, port=port, reload=False)
