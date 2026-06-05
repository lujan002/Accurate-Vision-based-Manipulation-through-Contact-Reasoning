#!/usr/bin/env python3
"""
Interactive SAM2 perception demo aligned with contact_reasoning perception plan.

- Sources: webcam, video file (.mp4), JPEG folder, optional RealSense (if pyrealsense2 installed).
- SAM2VideoPredictor + temporal memory for file / directory sources (full forward propagate after lock).
- Webcam / RealSense stream: centroid + ``SAM2ImagePredictor`` each frame. Unreliable single-point
  updates **do not** auto-reprompt: the last locked mask is kept; press **``r``** for a full GUI reprompt.
  (Rolling ``SAM2VideoPredictor`` on every webcam frame is too heavy; file sources use true video memory.)

Run from the contact_reasoning directory (same as sam2_test_img.py). Do not modify sam2_test_img.py.

Space  : manipulation timestep (stub pose / EKF log). The window still repaints each frame so ``event_only`` does not freeze the UI during long video propagation.
Enter  : confirm mask during locking / reprompt.
, . or Left/Right arrows : cycle top-3 masks.
r      : force full reprompt.
ESC / q: quit.

File sources: lock object on frame 0 (video is seeked to start before you click).

If ``import torch`` seems to hang or errors under ``.../torch/distributed/_pycute``, the venv is likely on a
slow or power-managed USB drive: move the project/venv to an internal SSD, or set e.g.
``export PYTHONPYCACHEPREFIX=/tmp/sam2_pyc`` so bytecode is not written to the USB volume, then retry.
"""

from __future__ import annotations

import os

# Before importing PyTorch: reduce import-time filesystem churn on slow / removable volumes.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
_pyc = os.environ.get("PYTHONPYCACHEPREFIX")
if _pyc:
    try:
        os.makedirs(_pyc, exist_ok=True)
    except OSError:
        pass

import argparse
import glob
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

print("Loading PyTorch and SAM2 (first run can be slow on USB/external disks; wait or use SSD)…", flush=True)
try:
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
except KeyboardInterrupt:
    raise SystemExit(
        "Import interrupted. PyTorch walks thousands of files on first import; on a slow or "
        "sleeping USB volume this can stall in torch.distributed. Fix: move the repo/.venv to an "
        "internal SSD, or run:  export PYTHONPYCACHEPREFIX=/tmp/sam2_pyc  "
        "then retry (keeps .pyc off the external disk)."
    ) from None

OBJ_ID = 1


class LockState(Enum):
    NEED_CLICK = auto()
    CHOOSE_MASK = auto()
    TRACKING = auto()
    REPROMPT = auto()


@dataclass
class TrackingHealthConfig:
    min_area_frac: float = 1e-4
    max_area_frac: float = 0.95
    min_iou_vs_prev: float = 0.04


def _mask_area_frac(mask: np.ndarray) -> float:
    m = mask.astype(bool)
    return float(m.mean())


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def _mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    m = mask.astype(bool)
    if not m.any():
        return None
    ys, xs = np.where(m)
    return float(xs.mean()), float(ys.mean())


def overlay_mask_bgr(
    bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int] = (255, 144, 30),
    alpha: float = 0.45,
) -> np.ndarray:
    m = (mask > 0.5).astype(np.float32)
    out = bgr.astype(np.float32)
    color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    blend = out * (1 - m[..., None] * alpha) + color * (m[..., None] * alpha)
    return np.clip(blend, 0, 255).astype(np.uint8)


def composite_status(bgr: np.ndarray, lines: List[str]) -> np.ndarray:
    out = bgr.copy()
    y0 = 24
    for i, line in enumerate(lines):
        cv2.putText(out, line, (12, y0 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, line, (12, y0 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


class FrameSource(ABC):
    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return RGB uint8 HxWx3 or None."""

    @abstractmethod
    def close(self) -> None:
        pass


class WebcamSource(FrameSource):
    def __init__(self, device_id: int = 0, width: int = 0, height: int = 0):
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam {device_id}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Optional[np.ndarray]:
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


class VideoFileSource(FrameSource):
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video {path}")

    def seek(self, frame_idx: int) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))

    def read(self) -> Optional[np.ndarray]:
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


class ImageSequenceSource(FrameSource):
    def __init__(self, directory: str):
        exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
        paths: List[str] = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(directory, e)))

        def sort_key(p: str) -> Tuple[int, str]:
            stem = Path(p).stem
            digits = "".join(ch for ch in stem if ch.isdigit())
            return (int(digits) if digits else 0, p)

        paths.sort(key=sort_key)
        if not paths:
            raise RuntimeError(f"No images in {directory}")
        self.paths = paths
        self._idx = 0

    def seek(self, idx: int) -> None:
        self._idx = int(idx) % len(self.paths)

    def read(self) -> Optional[np.ndarray]:
        p = self.paths[self._idx]
        return np.array(Image.open(p).convert("RGB"))

    def close(self) -> None:
        pass


class RealSenseSource(FrameSource):
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError as e:
            raise RuntimeError("pyrealsense2 not installed; use webcam or file source") from e
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(cfg)

    def read(self) -> Optional[np.ndarray]:
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return None
        bgr = np.asanyarray(color.get_data())
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.pipeline.stop()


def build_frame_source(source: str, path: Optional[str], webcam_id: int) -> FrameSource:
    if source == "webcam":
        return WebcamSource(webcam_id)
    if source == "realsense":
        return RealSenseSource()
    if source == "video":
        if not path:
            raise ValueError("video source requires --path")
        return VideoFileSource(path)
    if source == "images":
        if not path:
            raise ValueError("images source requires --path")
        return ImageSequenceSource(path)
    raise ValueError(f"Unknown source {source}")


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


@dataclass
class InteractiveSession:
    device: torch.device
    image_predictor: SAM2ImagePredictor
    video_predictor: SAM2VideoPredictor
    frame_source: FrameSource
    source_kind: str
    health_cfg: TrackingHealthConfig = field(default_factory=TrackingHealthConfig)

    lock_state: LockState = LockState.NEED_CLICK
    pending_click_xy: Optional[Tuple[int, int]] = None
    multimask_idx: int = 0
    candidate_masks: Optional[np.ndarray] = None
    candidate_scores: Optional[np.ndarray] = None
    lock_prompt_pts: Optional[np.ndarray] = None  # (N,2) float64 image pixels
    lock_prompt_labels: Optional[np.ndarray] = None  # (N,) int64

    locked_mask: Optional[np.ndarray] = None
    prev_mask: Optional[np.ndarray] = None
    video_inference_state: Optional[dict] = None
    video_masks_by_frame: Optional[Dict[int, np.ndarray]] = None
    video_frame_count: int = 0
    file_frame_index: int = 0
    last_bgr_display: Optional[np.ndarray] = None

    # After lock, skip one IoU check so we do not compare centroid refinement to the multimask.
    _skip_iou_once: bool = False
    # Webcam/realsense: first TRACKING frame only redraws locked mask (no centroid predict) to avoid
    # immediate begin_reprompt() when the first refine fails health vs the GUI-chosen mask.
    _stream_display_locked_only_once: bool = False

    def __post_init__(self) -> None:
        self._maybe_autocast = nullcontext()
        if self.device.type == "cuda":
            self._maybe_autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    def begin_reprompt(self) -> None:
        self.lock_state = LockState.NEED_CLICK
        self.pending_click_xy = None
        self.candidate_masks = None
        self.candidate_scores = None
        self.lock_prompt_pts = None
        self.lock_prompt_labels = None
        self.locked_mask = None
        self.prev_mask = None
        self.video_masks_by_frame = None
        self._skip_iou_once = False
        self._stream_display_locked_only_once = False

    def reset_video_file_state(self, video_path: str) -> None:
        with torch.inference_mode(), self._maybe_autocast:
            self.video_inference_state = self.video_predictor.init_state(
                video_path,
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )
        self.video_frame_count = int(self.video_inference_state["num_frames"])

    def run_full_propagation_file(
        self,
        frame_idx: int,
        points_xy: np.ndarray,
        labels: np.ndarray,
        *,
        progress_win: Optional[str] = None,
        progress_bgr: Optional[np.ndarray] = None,
    ) -> None:
        assert self.video_inference_state is not None
        with torch.inference_mode(), self._maybe_autocast:
            self.video_predictor.reset_state(self.video_inference_state)
            self.video_predictor.add_new_points_or_box(
                self.video_inference_state,
                frame_idx=frame_idx,
                obj_id=OBJ_ID,
                points=torch.tensor(points_xy, dtype=torch.float32),
                labels=torch.tensor(labels, dtype=torch.int32),
                clear_old_points=True,
                normalize_coords=True,
                box=None,
            )
            masks: Dict[int, np.ndarray] = {}
            for f_idx, _obj_ids, video_res_masks in self.video_predictor.propagate_in_video(
                self.video_inference_state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=None,
                reverse=False,
            ):
                m = video_res_masks[0, 0].detach().cpu().numpy() > 0.5
                masks[int(f_idx)] = m
                if progress_win is not None:
                    base = progress_bgr
                    if base is None:
                        base = np.zeros((480, 640, 3), dtype=np.uint8)
                    vis = composite_status(
                        base,
                        [
                            "Propagating SAM2 video memory…",
                            f"frames cached: {len(masks)} / {self.video_frame_count} (idx {int(f_idx)})",
                        ],
                    )
                    cv2.imshow(progress_win, vis)
                    cv2.waitKey(1)
        self.video_masks_by_frame = masks

    def image_reprompt_multimask(self, rgb: np.ndarray, points_xy: np.ndarray, labels: np.ndarray) -> None:
        with torch.inference_mode(), self._maybe_autocast:
            self.image_predictor.set_image(rgb)
            masks, scores, _ = self.image_predictor.predict(
                point_coords=points_xy,
                point_labels=labels,
                box=None,
                mask_input=None,
                multimask_output=True,
                return_logits=False,
                normalize_coords=True,
            )
            order = np.argsort(scores)[::-1]
            self.candidate_masks = masks[order]
            self.candidate_scores = scores[order]
            self.multimask_idx = 0

    def current_chosen_mask(self) -> Optional[np.ndarray]:
        if self.candidate_masks is None:
            return None
        return self.candidate_masks[self.multimask_idx]

    def tracking_health_ok(self, mask: np.ndarray, iou_ref: Optional[np.ndarray] = None) -> bool:
        """iou_ref: previous accepted mask for IoU; None skips IoU (e.g. first frame after lock)."""
        af = _mask_area_frac(mask)
        if af < self.health_cfg.min_area_frac or af > self.health_cfg.max_area_frac:
            return False
        if _mask_centroid(mask) is None:
            return False
        if iou_ref is not None and iou_ref.shape == mask.shape:
            if _mask_iou(mask, iou_ref) < self.health_cfg.min_iou_vs_prev:
                return False
        return True

    def on_tracking_tick_stream(self, rgb: np.ndarray) -> np.ndarray:
        """Centroid + image predictor between frames. Never calls begin_reprompt: single-point
        re-segmentation is unreliable; keep the last locked mask on failure and use ``r`` to re-pick."""
        h, w = rgb.shape[:2]
        if self.locked_mask is None:
            return np.zeros((h, w), dtype=bool)
        if self.locked_mask.shape != (h, w):
            self.locked_mask = (
                cv2.resize(self.locked_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
            )
        if self._stream_display_locked_only_once:
            self._stream_display_locked_only_once = False
            return self.locked_mask.copy()

        centroid = _mask_centroid(self.locked_mask)
        if centroid is None:
            logging.warning("stream track: no centroid on locked mask; keeping previous mask")
            return self.locked_mask.copy()

        pts = np.array([[centroid[0], centroid[1]]], dtype=np.float64)
        lbl = np.array([1], dtype=np.int64)
        try:
            with torch.inference_mode(), self._maybe_autocast:
                self.image_predictor.set_image(rgb)
                masks, _scores, _ = self.image_predictor.predict(
                    point_coords=pts,
                    point_labels=lbl,
                    multimask_output=False,
                    normalize_coords=True,
                )
            mask = masks[0] > 0.5
        except Exception:
            logging.exception("stream track: predict failed; keeping previous mask")
            return self.locked_mask.copy()

        if not mask.any():
            logging.debug("stream track: empty mask; keeping previous mask")
            return self.locked_mask.copy()

        iou_ref = None if self._skip_iou_once else self.locked_mask.copy()
        if not self.tracking_health_ok(mask, iou_ref=iou_ref):
            logging.debug("stream track: health rejected update; keeping previous mask")
            return self.locked_mask.copy()

        self._skip_iou_once = False
        self.prev_mask = self.locked_mask.copy()
        self.locked_mask = mask
        return mask

    def on_tracking_tick_file(self) -> Optional[np.ndarray]:
        if self.video_masks_by_frame is None:
            return None
        idx = self.file_frame_index % max(1, self.video_frame_count)
        return self.video_masks_by_frame.get(idx)

    def on_manipulation_timestep_stub(self, rgb: np.ndarray, mask: np.ndarray) -> None:
        c = _mask_centroid(mask)
        logging.info(
            "[manipulation_timestep] stub pose/EKF/affordances — centroid=%s mask_pixels=%.0f shape=%s",
            c,
            float(mask.sum()),
            rgb.shape[:2],
        )


def _poll_key() -> int:
    """Return raw key code (use waitKeyEx when available so arrow keys work)."""
    if hasattr(cv2, "waitKeyEx"):
        return int(cv2.waitKeyEx(1))
    return int(cv2.waitKey(1))


def _is_arrow_left(k: int) -> bool:
    # waitKeyEx (GTK/Qt): 65361; some builds use 63234 in the low 16 bits
    return k in (65361, 63234) or (k & 0xFFFF) in (63234, 65361)


def _is_arrow_right(k: int) -> bool:
    return k in (65363, 63235) or (k & 0xFFFF) in (63235, 65363)


def _sync_file_frame(source: FrameSource, frame_idx: int) -> None:
    if isinstance(source, VideoFileSource):
        source.seek(frame_idx)
    elif isinstance(source, ImageSequenceSource):
        source.seek(frame_idx)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    here = Path(__file__).resolve().parent
    os.chdir(here)

    ap = argparse.ArgumentParser(description="Interactive SAM2 tracking (see module docstring).")
    ap.add_argument("--source", choices=["webcam", "video", "images", "realsense"], default="webcam")
    ap.add_argument("--path", type=str, default=None, help="Video file or image directory")
    ap.add_argument("--webcam-id", type=int, default=0)
    ap.add_argument(
        "--preview",
        choices=["live", "event_only", "auto"],
        default="live",
        help="Reserved for future tiering; the UI always repaints so long video propagation can show progress.",
    )
    ap.add_argument("--checkpoint", default="sam2_repo/checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info("device=%s", device)

    ac = nullcontext()
    if device.type == "cuda":
        ac = torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    image_model = build_sam2(args.model_cfg, args.checkpoint, device=device.type)
    image_predictor = SAM2ImagePredictor(image_model)
    video_model = build_sam2_video_predictor(args.model_cfg, args.checkpoint, device=device.type)
    video_predictor: SAM2VideoPredictor = video_model  # type: ignore[assignment]

    frame_source = build_frame_source(args.source, args.path, args.webcam_id)

    session = InteractiveSession(
        device=device,
        image_predictor=image_predictor,
        video_predictor=video_predictor,
        frame_source=frame_source,
        source_kind=args.source,
    )

    video_disk_path: Optional[str] = None
    if args.source in ("video", "images"):
        if not args.path:
            ap.error("--path required for video/images")
        video_disk_path = os.path.abspath(args.path)

    win = "SAM2 interactive"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and session.lock_state in (
            LockState.NEED_CLICK,
            LockState.CHOOSE_MASK,
        ):
            session.pending_click_xy = (x, y)

    cv2.setMouseCallback(win, on_mouse)

    try:
        while True:
            is_file = args.source in ("video", "images")

            if is_file and session.lock_state != LockState.TRACKING:
                _sync_file_frame(frame_source, 0)
                session.file_frame_index = 0
            elif is_file and session.lock_state == LockState.TRACKING:
                _sync_file_frame(frame_source, session.file_frame_index)

            rgb = frame_source.read()
            if rgb is None and is_file:
                session.file_frame_index = 0
                _sync_file_frame(frame_source, 0)
                rgb = frame_source.read()
            if rgb is None:
                logging.warning("No frame")
                break

            h, w = rgb.shape[:2]
            mask_disp = np.zeros((h, w), dtype=bool)
            status = [
                f"{args.source} lock={session.lock_state.name} preview={args.preview}",
                "click=point , . or arrows=masks Enter=lock Space=plan r=reprompt q=quit",
            ]

            if session.lock_state == LockState.TRACKING:
                if args.source in ("webcam", "realsense"):
                    mask_disp = session.on_tracking_tick_stream(rgb)
                else:
                    # Trust SAM2VideoPredictor masks for file sources. IoU vs the GUI multimask or
                    # occasional missing dict keys caused false ``begin_reprompt`` after a fixed grace.
                    mf = session.on_tracking_tick_file()
                    fresh_prop = False
                    if mf is not None:
                        if mf.shape != (h, w):
                            mf = cv2.resize(mf.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                        mask_disp = mf
                        fresh_prop = True
                    elif session.locked_mask is not None and session.locked_mask.shape == (h, w):
                        mask_disp = session.locked_mask.copy()
                    if fresh_prop:
                        if mask_disp.any():
                            session._skip_iou_once = False
                            session.prev_mask = (
                                session.locked_mask.copy() if session.locked_mask is not None else None
                            )
                            session.locked_mask = mask_disp.copy()
                        else:
                            logging.warning(
                                "file track: empty propagated mask at frame %s; holding last mask",
                                session.file_frame_index,
                            )
                            if session.locked_mask is not None and session.locked_mask.shape == (h, w):
                                mask_disp = session.locked_mask.copy()

            elif session.lock_state in (LockState.NEED_CLICK, LockState.CHOOSE_MASK):
                if session.pending_click_xy is not None:
                    x, y = session.pending_click_xy
                    pts = np.array([[x, y]], dtype=np.float64)
                    lbl = np.array([1], dtype=np.int64)
                    session.lock_prompt_pts = pts.copy()
                    session.lock_prompt_labels = lbl.copy()
                    session.image_reprompt_multimask(rgb, pts, lbl)
                    session.pending_click_xy = None
                    session.lock_state = LockState.CHOOSE_MASK
                if session.lock_state == LockState.CHOOSE_MASK and session.candidate_masks is not None:
                    cm = session.current_chosen_mask()
                    if cm is not None:
                        mask_disp = cm

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            overlay = overlay_mask_bgr(bgr, mask_disp)
            vis = composite_status(overlay, status)
            session.last_bgr_display = vis

            cv2.imshow(win, vis)

            k = _poll_key()
            key = k & 0xFF

            if key in (27, ord("q")):
                break
            if key == ord("r"):
                session.begin_reprompt()
                logging.info("User reprompt")

            cycle_left = key in (ord(","),) or _is_arrow_left(k)
            cycle_right = key in (ord("."),) or _is_arrow_right(k)
            if session.lock_state == LockState.CHOOSE_MASK and session.candidate_masks is not None:
                if cycle_left:
                    session.multimask_idx = (session.multimask_idx - 1) % 3
                elif cycle_right:
                    session.multimask_idx = (session.multimask_idx + 1) % 3

            if key in (13, 10) and session.lock_state == LockState.CHOOSE_MASK:
                if session.candidate_masks is None or session.lock_prompt_pts is None:
                    continue
                session.locked_mask = session.current_chosen_mask()
                if session.locked_mask is None:
                    continue
                session.prev_mask = None
                session.lock_state = LockState.TRACKING
                session.candidate_masks = None
                session.candidate_scores = None
                session._skip_iou_once = True
                if args.source in ("webcam", "realsense"):
                    session._stream_display_locked_only_once = True

                if video_disk_path is not None and session.lock_prompt_pts is not None:
                    logging.info("Video propagate on %s …", video_disk_path)
                    session.reset_video_file_state(video_disk_path)
                    bgr_snap = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    session.run_full_propagation_file(
                        0,
                        session.lock_prompt_pts,
                        session.lock_prompt_labels,
                        progress_win=win,
                        progress_bgr=bgr_snap,
                    )
                    session.file_frame_index = 0
                    _sync_file_frame(frame_source, 0)
                logging.info("Mask locked; tracking.")

            if key == ord(" "):
                m = session.locked_mask if session.lock_state == LockState.TRACKING else mask_disp
                if m is not None and m.any():
                    session.on_manipulation_timestep_stub(rgb, m)

            if session.lock_state == LockState.TRACKING and is_file and session.video_masks_by_frame is not None:
                session.file_frame_index = (session.file_frame_index + 1) % max(1, session.video_frame_count)

    finally:
        frame_source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
