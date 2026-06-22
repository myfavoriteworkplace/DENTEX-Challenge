import os
import sys
import json
import logging
import glob
from pathlib import Path

from detectron2.config import get_cfg
from hierarchialdet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from hierarchialdet.util.model_ema import (
    add_model_ema_configs,
    may_build_model_ema,
    may_get_ema_checkpointer,
    EMAHook,
    apply_model_ema_and_restore,
    EMADetectionCheckpointer,
)
from hierarchialdet.predictor import VisualizationDemo
import argparse
import SimpleITK as sitk

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("hierarchialdet.process")


def load_image_index(index_path: str) -> dict:
    """Load image metadata from JSON and return a filename→id mapping for O(1) lookup."""
    index_path = str(index_path)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Image index file not found: {index_path}")
    with open(index_path) as f:
        entries = json.load(f)
    return {entry["file_name"]: entry["id"] for entry in entries}


def custom_format_output(outputs, img_ids):
    boxes = []
    for k, instances in enumerate(outputs):
        for i in range(len(instances)):
            instance = instances[i]
            bbox_coords = instance.pred_boxes.tensor[0].tolist()
            category_id_1 = instance.pred_classes_1[0].item()
            category_id_2 = instance.pred_classes_2[0].item()
            img_id = img_ids[k]
            box = {
                "name": f"{category_id_1} - {category_id_2}",
                "corners": [
                    [bbox_coords[0], bbox_coords[1], img_id],
                    [bbox_coords[0], bbox_coords[3], img_id],
                    [bbox_coords[2], bbox_coords[1], img_id],
                    [bbox_coords[2], bbox_coords[3], img_id],
                ],
                "probability": instance.scores[0].item(),
            }
            boxes.append(box)

    return {
        "name": "Regions of interest",
        "type": "Multiple 2D bounding boxes",
        "boxes": boxes,
        "version": {"major": 1, "minor": 0},
    }


def coco_format_output(outputs, img_ids):
    coco_annotations = []
    for k, instances in enumerate(outputs):
        for i in range(len(instances)):
            instance = instances[i]
            bbox_coords = instance.pred_boxes.tensor[0].tolist()
            bbox_coords[2] = bbox_coords[2] - bbox_coords[0]
            bbox_coords[3] = bbox_coords[3] - bbox_coords[1]
            coco_annotation = {
                "image_id": img_ids[k],
                "category_id_1": instance.pred_classes_1[0].item(),
                "category_id_2": instance.pred_classes_2[0].item(),
                "bbox": bbox_coords,
                "score": instance.scores[0].item(),
            }
            coco_annotations.append(coco_annotation)
    return coco_annotations


def get_parser():
    parser = argparse.ArgumentParser(
        description="HierarchialDet initial phase inference for dental X-ray analysis"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.0")),
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--nclass",
        type=int,
        default=2,
        help="Number of trained class levels (1=quadrant, 2=+enumeration)",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class Hierarchialdet:
    def __init__(self):
        self.cfg = None
        self.demo = None
        self.config_path = os.environ.get(
            "CONFIG_PATH",
            "/opt/app/configs/diffdet.custom.swinbase.nonpretrain.yaml",
        )
        self.weights_path = os.environ.get(
            "MODEL_WEIGHTS",
            "/opt/app/pretrained_model/model_final.pth",
        )
        self.input_dir = os.environ.get(
            "INPUT_DIR",
            "/input/images/panoramic-dental-xrays",
        )
        self.output_path = os.environ.get(
            "OUTPUT_PATH",
            "/output/abnormal-teeth-detection.json",
        )
        self.index_path = os.environ.get(
            "IMAGE_INDEX_PATH",
            str(Path(__file__).parent / "val_ids.json"),
        )

    def validate_inputs(self):
        """Validate all required files and directories exist before running inference."""
        errors = []

        if not os.path.isfile(self.config_path):
            errors.append(f"Config file not found: {self.config_path}")

        if not os.path.isfile(self.weights_path):
            errors.append(f"Model weights not found: {self.weights_path}")

        mha_files = glob.glob(os.path.join(self.input_dir, "*.mha"))
        if not mha_files:
            errors.append(f"No .mha input files found in: {self.input_dir}")

        if errors:
            for msg in errors:
                logger.error("Validation error: %s", msg)
            raise FileNotFoundError(
                f"Input validation failed with {len(errors)} error(s). See logs above."
            )

        logger.info("Input validation passed. Found %d .mha file(s).", len(mha_files))
        return mha_files

    def setup(self):
        args = get_parser().parse_args()
        logger.info("Setting up model configuration from: %s", self.config_path)
        self.cfg = get_cfg()
        add_diffusiondet_config(self.cfg)
        add_model_ema_configs(self.cfg)
        self.cfg.merge_from_file(self.config_path)
        self.cfg.MODEL.WEIGHTS = self.weights_path
        self.cfg.merge_from_list(args.opts)
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
            args.confidence_threshold
        )
        self.cfg.freeze()
        logger.info("Loading model weights from: %s", self.weights_path)
        self.demo = VisualizationDemo(self.cfg, k=1)
        logger.info("Model ready.")

    def process(self):
        try:
            mha_files = self.validate_inputs()
        except FileNotFoundError as exc:
            logger.error("Aborting: %s", exc)
            sys.exit(1)

        self.setup()

        try:
            image_index = load_image_index(self.index_path)
            logger.info(
                "Loaded image index with %d entries from %s",
                len(image_index),
                self.index_path,
            )
        except FileNotFoundError as exc:
            logger.warning(
                "Image index unavailable, slice indices will default to slice number. %s",
                exc,
            )
            image_index = {}

        all_outputs = []
        img_ids = []

        file_path = mha_files[0]
        logger.info("Reading input file: %s", file_path)

        try:
            image = sitk.ReadImage(file_path)
        except RuntimeError as exc:
            logger.error("Failed to read input image '%s': %s", file_path, exc)
            sys.exit(1)

        image_array = sitk.GetArrayFromImage(image)
        total_slices = image_array.shape[2]
        logger.info("Processing %d image slice(s).", total_slices)

        for k in range(total_slices):
            image_name = f"val_{k}.png"
            logger.info("Inference on slice %d/%d (%s)", k + 1, total_slices, image_name)

            try:
                predictions, _ = self.demo.run_on_image(image_array[:, :, k, :])
            except Exception as exc:
                logger.error("Inference failed on slice %s: %s", image_name, exc)
                raise

            instances = predictions["instances"]
            all_outputs.append(instances)

            img_id = image_index.get(image_name)
            if img_id is None:
                logger.warning(
                    "No ID found for '%s' in image index. Defaulting to slice index %d.",
                    image_name,
                    k,
                )
                img_id = k
            img_ids.append(img_id)

        logger.info("All slices processed. Formatting output.")
        annotations = custom_format_output(all_outputs, img_ids)

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_path, "w") as f:
            json.dump(annotations, f, indent=2)

        logger.info("Results written to %s", self.output_path)


if __name__ == "__main__":
    Hierarchialdet().process()
