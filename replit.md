# DENTEX 2023 Challenge - HierarchicalDet Baseline

## Project Overview

This repository contains the baseline implementation for the **DENTEX 2023 Challenge** (Dental Enumeration and Diagnosis on Panoramic X-rays), held at MICCAI 2023. It implements **HierarchicalDet** — a diffusion-based hierarchical multi-label object detection model for analyzing panoramic dental X-rays.

The model detects abnormal teeth and outputs bounding boxes with quadrant (Q), enumeration (N), and diagnosis (D) labels using the FDI numbering system.

## Project Structure

- **`HierarchialDet-FinalPhase-Docker/`** — Final phase: Abnormal tooth detection with diagnosis (main inference code)
- **`HierarchialDet-InitialPhase-Docker/`** — Initial phase: Tooth detection and enumeration
- **`DentexChallenge-AlgorithmPhases-Evaluation-Docker/`** — Evaluation scripts for benchmarking
- **`figures/`** — Documentation figures
- **`main.py`** — Entry point: environment check and project overview

## Running

The workflow runs `python main.py` which verifies the environment and shows usage instructions.

### Inference (requires pretrained weights + input images)

```bash
# Final phase (abnormal tooth detection + diagnosis)
cd HierarchialDet-FinalPhase-Docker && python -m process

# Initial phase (tooth detection + enumeration)
cd HierarchialDet-InitialPhase-Docker && python -m process
```

### Docker (original deployment method)

```bash
cd HierarchialDet-FinalPhase-Docker
bash build.sh   # Build container
bash test.sh    # Run inference test
bash export.sh  # Export for challenge submission
```

## Dependencies

Core Python packages (installed via pip --user):
- PyTorch + torchvision (ML framework)
- Detectron2 (object detection, bundled in each Docker dir)
- OpenCV, SimpleITK, Pillow (image processing)
- NumPy, SciPy, Pandas (data handling)
- pycocotools, fvcore, timm, einops (ML utilities)

## Data

Dataset available on Hugging Face: https://huggingface.co/datasets/ibrahimhamamci/DENTEX

## User Preferences

- Keep existing Docker-based project structure intact
