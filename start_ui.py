#!/usr/bin/env python3
"""Start the packaged UI using the active Python environment."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    entry = Path(__file__).resolve().parent / 'verified_meniscus_and_cartilage_models_2026-11-11' / 'segmentation_ui.py'
    runpy.run_path(str(entry), run_name='__main__')
