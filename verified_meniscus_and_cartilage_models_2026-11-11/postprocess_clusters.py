#!/usr/bin/env python3
"""Entry point for the shared postprocessor; prepare with --labels 1 2 3 4."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent.parent /
                       "segmentation_postprocessing" / "postprocess_clusters.py"), run_name="__main__")
