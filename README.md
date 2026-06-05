# contact_reasoning

Scripts and assets for SAM 2–based interactive tracking experiments.

## Prerequisites

1. **SAM 2** — this directory expects a sibling clone named `sam2_repo`:

   ```bash
   git clone https://github.com/facebookresearch/sam2.git sam2_repo
   ```

   Follow [SAM 2 INSTALL.md](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) to install dependencies and download checkpoints into `sam2_repo/checkpoints/`.

2. **Python** — use a virtual environment in this folder (for example `.venv`). The `.gitignore` excludes local venvs from version control.

## Layout

- `sam2_interactive_track.py`, `sam2_test_img.py` — project scripts
- `media/` — sample images and videos for testing

`sam2_repo` is intentionally not committed here (nested repository and large model weights).
