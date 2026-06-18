import sys
import os

def print_banner():
    print("=" * 60)
    print("  DENTEX 2023 Challenge - HierarchicalDet Baseline")
    print("  Dental Enumeration & Diagnosis on Panoramic X-rays")
    print("=" * 60)
    print()

def check_environment():
    print("Environment check:")
    print(f"  Python version: {sys.version.split()[0]}")

    checks = {
        "numpy": False,
        "PIL (Pillow)": False,
        "cv2 (OpenCV)": False,
        "SimpleITK": False,
        "pandas": False,
        "matplotlib": False,
        "scipy": False,
        "pycocotools": False,
        "omegaconf": False,
        "seaborn": False,
    }

    try:
        import numpy
        checks["numpy"] = True
    except ImportError:
        pass

    try:
        import PIL
        checks["PIL (Pillow)"] = True
    except ImportError:
        pass

    try:
        import cv2
        checks["cv2 (OpenCV)"] = True
    except ImportError:
        pass

    try:
        import SimpleITK
        checks["SimpleITK"] = True
    except ImportError:
        pass

    try:
        import pandas
        checks["pandas"] = True
    except ImportError:
        pass

    try:
        import matplotlib
        checks["matplotlib"] = True
    except ImportError:
        pass

    try:
        import scipy
        checks["scipy"] = True
    except ImportError:
        pass

    try:
        import pycocotools
        checks["pycocotools"] = True
    except ImportError:
        pass

    try:
        import omegaconf
        checks["omegaconf"] = True
    except ImportError:
        pass

    try:
        import seaborn
        checks["seaborn"] = True
    except ImportError:
        pass

    all_ok = True
    for name, ok in checks.items():
        status = "OK" if ok else "MISSING"
        print(f"  [{status:7}] {name}")
        if not ok:
            all_ok = False

    return all_ok

def show_project_structure():
    print()
    print("Project components:")
    components = [
        ("HierarchialDet-FinalPhase-Docker/", "Final phase: Abnormal tooth detection + diagnosis"),
        ("HierarchialDet-InitialPhase-Docker/", "Initial phase: Tooth detection and enumeration"),
        ("DentexChallenge-AlgorithmPhases-Evaluation-Docker/", "Evaluation scripts for challenge benchmarking"),
        ("figures/", "Documentation figures"),
    ]
    for path, desc in components:
        exists = os.path.exists(path)
        marker = "+" if exists else "-"
        print(f"  [{marker}] {path}")
        print(f"       {desc}")

    print()
    print("Usage:")
    print("  Inference (Final Phase):")
    print("    cd HierarchialDet-FinalPhase-Docker && python -m process")
    print()
    print("  Inference (Initial Phase):")
    print("    cd HierarchialDet-InitialPhase-Docker && python -m process")
    print()
    print("  Docker build (Final Phase):")
    print("    cd HierarchialDet-FinalPhase-Docker && bash build.sh")
    print()
    print("Note: Inference requires pretrained model weights and input X-ray images.")
    print("      See README.md or https://huggingface.co/datasets/ibrahimhamamci/DENTEX")

def main():
    print_banner()
    all_ok = check_environment()
    show_project_structure()
    print()
    if all_ok:
        print("All core dependencies are available. Ready to run inference.")
    else:
        print("Some dependencies are missing.")
        print("Install with: pip install --user -r HierarchialDet-FinalPhase-Docker/requirements.txt")
    print()

if __name__ == "__main__":
    main()
