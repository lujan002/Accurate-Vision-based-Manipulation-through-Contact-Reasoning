#!/usr/bin/env python3
"""
Interactive SAM2 perception demo aligned with contact_reasoning perception plan.

- Sources: webcam, video file (.mp4), JPEG folder, optional RealSense (if pyrealsense2 installed).
- SAM2VideoPredictor + temporal memory for file / directory sources (full forward propagate after lock).
- Webcam / RealSense stream: ``SAM2VideoPredictor`` with an in-memory frame buffer — each new frame is
  appended, then ``propagate_in_video`` runs for that index only so mask memory carries across frames.
  Optional ``--stream-video-state-cap`` re-anchors with ``add_new_mask`` to cap GPU memory on long runs.
  Locking still uses ``SAM2ImagePredictor`` multimasks from your click. Press **``r``** for a full reprompt.
  **Temporal memory:** later frames are conditioned via encoded memory from past frames (not by feeding the
  previous binary mask as a direct ``mask_input`` each tick the way the image predictor can).

``--preview event_only`` : for **webcam / RealSense** while **TRACKING**, the **overlay** refreshes only when
  you press **Space** (tracking still runs every frame so Space samples the latest mask). Ignored for
  video / image-folder sources (overlay stays live so the mask matches the seeked frame).

Space  : manipulation timestep (stub pose / EKF log). In ``event_only`` + **TRACKING** (stream sources),
  also publishes the latest mask to the display overlay.
**Default** ``--lock-on click``: one left-click runs multimask then **locks the top mask and starts tracking**
  immediately. ``--lock-on enter``: click only sets the point; use ``,`` / ``.`` or Left/Right arrows to cycle
  the top-3 masks, then **Enter** to lock. **Left-click while tracking** (webcam / RealSense) discards stream
  memory and re-picks the same way (click vs enter applies to that repick too).
r      : force full reprompt (clears file + stream state).

``--preview live`` / ``auto`` : overlay follows ``mask_disp`` every frame. ``event_only`` freezes the
  overlay until **Space** (tracking still runs in the background).
**Display layout:** OpenCV window is ~1.5× frame width — main view left; right column (half the frame width)
  stacks two panels. When ``--top-down on`` and the green quad is set, the **lower** half shows the
  **reprojected masked object**: the four clicks are only a *template* mapping the quad to a rectangle
  whose aspect is **metrically recovered** from the corners (the camera focal length is estimated from the
  four points; principal point assumed at image center), falling back to the quad edge ratio for near-frontal
  views. That single homography is applied to the whole frame, so the segmentation mask is reprojected
  correctly wherever it sits — it need not overlap the green quad. The plan-space bounding box of the mask
  is then auto-framed (letterboxed) into the panel,
  so no expand factor is needed. The **upper** half shows an **oriented min-area rectangle** warp of the
  mask (image-plane upright crop, not the table model). When ``--top-down off`` (or before the quad exists),
  the top panel follows ``--bb-orientation``: ``aligned`` = ``cv2.boundingRect`` crop; ``free`` =
  ``minAreaRect`` warp. Four clicks define the table quad (TL, TR, BR, BL in the image). ``t`` clears
  calibration. ``--top-down off``: lower panel black.
  **Note:** Anything not on the table plane (keyboard height, hands, walls) will still look perspective in
  the plan panel; only the table patch is modeled.
ESC / q: quit.

Run from the ``contact_reasoning`` directory (same as ``sam2_test_img.py``). Do not modify ``sam2_test_img.py``.

File sources: lock object on frame 0 (video is seeked to start before you click).

If ``import torch`` seems to hang or errors under ``.../torch/distributed/_pycute``, the venv is likely on a
slow or power-managed USB drive: move the project/venv to an internal SSD, or set e.g.
``export PYTHONPYCACHEPREFIX=/tmp/sam2_pyc`` so bytecode is not written to the USB volume, then retry.

**Webcam performance:** temporal memory costs more per frame than the old centroid+image path. Use
``--webcam-buffer 1`` (default), optional ``--webcam-fps`` / ``--webcam-width`` / ``--webcam-height``,
``--webcam-device /dev/videoN`` (overrides ``--webcam-id``), ``--webcam-flush-grabs`` to reduce backlog,
and tune ``--stream-video-state-cap`` (lower cap = more frequent re-anchor, less GPU RAM for the frame tensor
stack). Effective FPS is usually capped by SAM inference time.
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
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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


@contextmanager
def _silence_sam2_video_tqdm() -> Iterator[None]:
    """Patch tqdm in ``sam2_video_predictor`` (env ``TQDM_DISABLE`` is often applied too late)."""
    import sam2.sam2_video_predictor as _s2vp

    _orig = _s2vp.tqdm

    def _quiet(iterable, *args, **kwargs):  # type: ignore[no-untyped-def]
        return iterable

    _s2vp.tqdm = _quiet  # type: ignore[assignment]
    try:
        yield
    finally:
        _s2vp.tqdm = _orig


def rgb_hwc_uint8_to_sam2_image_tensor(
    rgb: np.ndarray,
    image_size: int,
    compute_device: torch.device,
    offload_video_to_cpu: bool,
) -> torch.Tensor:
    """One RGB frame -> (3, image_size, image_size) float32, ImageNet-normalized (same as ``load_video_frames``)."""
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    img = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
    storage = torch.device("cpu") if offload_video_to_cpu else compute_device
    t = t.to(storage)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=storage)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=storage)[:, None, None]
    return (t - mean) / std


def init_video_state_from_images_tensor(
    predictor: SAM2VideoPredictor,
    images: torch.Tensor,
    video_height: int,
    video_width: int,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
) -> Dict[str, Any]:
    """
    Same structure as ``SAM2VideoPredictor.init_state`` but ``images`` is already normalized
    ``(T, 3, image_size, image_size)`` for streaming / in-memory clips.
    """
    compute_device = predictor.device
    inference_state: Dict[str, Any] = {}
    inference_state["images"] = images
    inference_state["num_frames"] = int(images.shape[0])
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    inference_state["video_height"] = int(video_height)
    inference_state["video_width"] = int(video_width)
    inference_state["device"] = compute_device
    inference_state["storage_device"] = (
        torch.device("cpu") if offload_state_to_cpu else compute_device
    )
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state


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


@dataclass
class StreamStabilityConfig:
    """Gate centroid+SAM stream updates so the mask does not snap to unrelated objects."""

    min_iou_vs_prev: float = 0.14
    anchor_overlap_start: float = 0.22
    anchor_overlap_end: float = 0.07
    anchor_overlap_decay_frames: int = 420
    anchor_dilate_ratio: float = 0.11
    max_centroid_step_frac: float = 0.065
    first_refine_centroid_step_mult: float = 2.2


def _centroid_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


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


def mask_to_shape_hw(mask: Optional[np.ndarray], h: int, w: int) -> np.ndarray:
    """Boolean mask resized to (h, w) for compositing on the current RGB frame."""
    if mask is None:
        return np.zeros((h, w), dtype=bool)
    m = mask.astype(bool)
    if m.shape == (h, w):
        return m
    return (cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5).astype(bool)


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


def _letterbox_bgr(src: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize ``src`` uniformly to fit inside ``out_w``×``out_h`` and center on a black canvas."""
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    if src.size == 0 or out_w < 1 or out_h < 1:
        return out
    sh, sw = src.shape[:2]
    if sw < 1 or sh < 1:
        return out
    scale = min(float(out_w) / float(sw), float(out_h) / float(sh))
    nw = max(1, int(round(sw * scale)))
    nh = max(1, int(round(sh * scale)))
    resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    y0 = (out_h - nh) // 2
    x0 = (out_w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _order_quad_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order ``cv2.boxPoints`` quad as top-left, top-right, bottom-right, bottom-left (float32)."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts.astype(np.float32), axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def oriented_min_area_rect_mask_patch_bgr(
    bgr: np.ndarray,
    mask_hw: np.ndarray,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """
    Largest contour of ``mask_hw`` → ``cv2.minAreaRect`` → perspective-warp the tight oriented
    rectangle to axis-aligned, then letterbox into ``out_w``×``out_h``. Empty mask → black image.
    """
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    if out_w < 1 or out_h < 1:
        return out
    if bgr.shape[:2] != mask_hw.shape[:2]:
        return out
    m = (mask_hw > 0).astype(np.uint8) * 255
    if not int(m.sum()):
        return out
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return out
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1.0:
        return out
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    quad = _order_quad_tl_tr_br_bl(box)
    wa = float(np.linalg.norm(quad[1] - quad[0]))
    hb = float(np.linalg.norm(quad[3] - quad[0]))
    if wa < 1.0 or hb < 1.0:
        return out
    iw = max(1, int(np.ceil(wa)))
    ih = max(1, int(np.ceil(hb)))
    dst = np.array([[0, 0], [iw - 1, 0], [iw - 1, ih - 1], [0, ih - 1]], dtype=np.float32)
    try:
        M = cv2.getPerspectiveTransform(quad, dst)
    except Exception:
        return out
    warped = cv2.warpPerspective(bgr, M, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return _letterbox_bgr(warped, out_w, out_h)


def axis_aligned_bbox_mask_patch_bgr(
    bgr: np.ndarray,
    mask_hw: np.ndarray,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """
    Largest contour of ``mask_hw`` → ``cv2.boundingRect`` (edges parallel to the image), crop ``bgr``,
    then letterbox into ``out_w``×``out_h``. Empty mask → black image.
    """
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    if out_w < 1 or out_h < 1:
        return out
    if bgr.shape[:2] != mask_hw.shape[:2]:
        return out
    m = (mask_hw > 0).astype(np.uint8) * 255
    if not int(m.sum()):
        return out
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return out
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1.0:
        return out
    x, y, rw, rh = cv2.boundingRect(cnt)
    if rw < 1 or rh < 1:
        return out
    H, W = bgr.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(W, int(x) + int(rw))
    y1 = min(H, int(y) + int(rh))
    if x1 <= x0 or y1 <= y0:
        return out
    crop = bgr[y0:y1, x0:x1]
    return _letterbox_bgr(crop, out_w, out_h)


def composite_main_and_side_panels_bgr(
    main_bgr: np.ndarray,
    crop_bgr: np.ndarray,
    mask_for_crop: np.ndarray,
    *,
    bb_orientation: str = "aligned",
    right_width_frac: float = 0.5,
    lower_panel_bgr: Optional[np.ndarray] = None,
    upper_panel_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Place ``main_bgr`` (HxW) on the left and a right column (~``right_width_frac``×W wide, H tall)
    with an upper panel and a lower panel (black, or ``lower_panel_bgr`` if it matches the expected size).

    If ``upper_panel_bgr`` matches ``(ph_top, strip_w)``, it is used as-is. Otherwise the upper panel is a
    letterboxed crop of ``crop_bgr`` (see ``bb_orientation``).
    """
    h, w = main_bgr.shape[:2]
    strip_w = max(1, int(round(float(w) * float(right_width_frac))))
    ph_top = h // 2
    ph_bot = h - ph_top
    if (
        upper_panel_bgr is not None
        and upper_panel_bgr.ndim == 3
        and upper_panel_bgr.shape[0] == ph_top
        and upper_panel_bgr.shape[1] == strip_w
    ):
        upper = upper_panel_bgr
    elif bb_orientation == "free":
        upper = oriented_min_area_rect_mask_patch_bgr(crop_bgr, mask_for_crop, strip_w, ph_top)
    else:
        upper = axis_aligned_bbox_mask_patch_bgr(crop_bgr, mask_for_crop, strip_w, ph_top)
    if (
        lower_panel_bgr is not None
        and lower_panel_bgr.ndim == 3
        and lower_panel_bgr.shape[0] == ph_bot
        and lower_panel_bgr.shape[1] == strip_w
    ):
        lower = lower_panel_bgr
    else:
        lower = np.zeros((ph_bot, strip_w, 3), dtype=np.uint8)
    right = np.vstack([upper, lower])
    return np.hstack([main_bgr, right])


def _quad_double_area(pts: np.ndarray) -> float:
    """Twice the signed polygon area of ordered points (4,2); used to reject degenerate quads."""
    p = pts.astype(np.float64)
    if p.shape != (4, 2):
        return 0.0
    s = 0.0
    for i in range(4):
        j = (i + 1) % 4
        s += p[i, 0] * p[j, 1] - p[j, 0] * p[i, 1]
    return float(s)


def _plan_rect_aspect_from_quad(src_quad_tl_tr_br_bl: np.ndarray) -> float:
    """
    Width/height aspect of the rectified rectangle implied by the calibration quad, using the **average of
    opposite edges** (more stable than a single edge): ``width = mean(|TR-TL|, |BR-BL|)`` and
    ``height = mean(|BL-TL|, |BR-TR|)``. Clamped to a sane range.

    This is only a template aspect; a 4-point homography removes perspective exactly regardless, so a
    slightly off aspect merely stretches the plan view uniformly (it never re-introduces a trapezoid look).
    """
    q = np.asarray(src_quad_tl_tr_br_bl, dtype=np.float64).reshape(4, 2)
    top = float(np.linalg.norm(q[1] - q[0]))
    bottom = float(np.linalg.norm(q[2] - q[3]))
    left = float(np.linalg.norm(q[3] - q[0]))
    right = float(np.linalg.norm(q[2] - q[1]))
    width = max(0.5 * (top + bottom), 1e-3)
    height = max(0.5 * (left + right), 1e-3)
    return float(np.clip(width / height, 0.05, 20.0))


def _recover_rect_aspect_metric(
    src_quad_tl_tr_br_bl: np.ndarray,
    image_w: int,
    image_h: int,
) -> Optional[float]:
    """
    True width/height aspect of the real-world rectangle from its perspective image (Zhang & He,
    "Whiteboard scanning and image enhancement"), or ``None`` when the geometry is degenerate.

    Estimates the camera focal length from the two vanishing points (principal point assumed at the image
    center), then recovers the rectangle aspect via the image of the absolute conic. Returns ``None`` for
    near-affine views (vanishing points at infinity) or invalid focal estimates, so the caller can fall
    back to ``_plan_rect_aspect_from_quad`` — which is accurate precisely in that near-affine regime.
    """
    eps = 1e-9
    q = np.asarray(src_quad_tl_tr_br_bl, dtype=np.float64).reshape(4, 2)
    u0 = 0.5 * float(image_w)
    v0 = 0.5 * float(image_h)
    # Principal point to origin; homogeneous. Ordering: m1=TL, m2=TR, m3=BL, m4=BR.
    m1 = np.array([q[0, 0] - u0, q[0, 1] - v0, 1.0])
    m2 = np.array([q[1, 0] - u0, q[1, 1] - v0, 1.0])
    m3 = np.array([q[3, 0] - u0, q[3, 1] - v0, 1.0])
    m4 = np.array([q[2, 0] - u0, q[2, 1] - v0, 1.0])

    c14 = np.cross(m1, m4)
    den_k2 = float(np.dot(np.cross(m2, m4), m3))
    den_k3 = float(np.dot(np.cross(m3, m4), m2))
    if abs(den_k2) < eps or abs(den_k3) < eps:
        return None
    k2 = float(np.dot(c14, m3)) / den_k2
    k3 = float(np.dot(c14, m2)) / den_k3

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    denom = n2[2] * n3[2]
    if abs(denom) < eps:
        return None
    f2 = -(n2[0] * n3[0] + n2[1] * n3[1]) / denom
    if not np.isfinite(f2) or f2 <= 0.0:
        return None

    num = (n2[0] ** 2 + n2[1] ** 2) / f2 + n2[2] ** 2
    dna = (n3[0] ** 2 + n3[1] ** 2) / f2 + n3[2] ** 2
    if not np.isfinite(num) or not np.isfinite(dna) or dna <= eps or num <= eps:
        return None
    ratio2 = num / dna
    if not np.isfinite(ratio2) or ratio2 <= 0.0:
        return None
    aspect = float(np.sqrt(ratio2))
    if not np.isfinite(aspect) or not (0.05 <= aspect <= 20.0):
        return None
    return aspect


def compute_table_topdown_homography(
    src_quad_tl_tr_br_bl: np.ndarray,
    *,
    image_w: int,
    image_h: int,
) -> Optional[np.ndarray]:
    """
    Homography H (3×3) mapping image pixels onto a metric **plan frame**, or None.

    The calibration quad is the *template* only: it is mapped to an axis-aligned rectangle whose aspect
    ratio is **metrically recovered** from the four corners (``_recover_rect_aspect_metric``, which
    estimates the camera focal length), falling back to the edge-length ratio
    (``_plan_rect_aspect_from_quad``) when the view is near-affine or the estimate is invalid. The
    rectangle is placed at the plan origin with a fixed base height; the absolute scale is irrelevant
    because the caller re-frames the plan onto the panel. Because H is a plane-to-plane homography it
    applies to **every** pixel, not just those inside the green quad, so the segmentation mask is
    reprojected correctly wherever it lies.
    """
    src = np.asarray(src_quad_tl_tr_br_bl, dtype=np.float32).reshape(4, 2)
    if abs(_quad_double_area(src)) < 8.0:
        return None
    aspect = _recover_rect_aspect_metric(src, image_w, image_h)
    if aspect is None:
        aspect = _plan_rect_aspect_from_quad(src)
    base_h = 1000.0
    base_w = max(1.0, aspect * base_h)
    dst = np.array(
        [[0.0, 0.0], [base_w, 0.0], [base_w, base_h], [0.0, base_h]],
        dtype=np.float32,
    )
    try:
        return cv2.getPerspectiveTransform(src, dst)
    except Exception:
        return None


def _plan_bbox_of_mask(H: np.ndarray, mask_hw: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """Plan-space (x0, y0, x1, y1) bounding box of the mask's nonzero pixels after applying ``H``."""
    ys, xs = np.where(mask_hw > 0)
    if xs.size == 0:
        return None
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1).reshape(-1, 1, 2)
    plan = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    finite = plan[np.isfinite(plan).all(axis=1)]
    if finite.shape[0] == 0:
        return None
    x0, y0 = float(finite[:, 0].min()), float(finite[:, 1].min())
    x1, y1 = float(finite[:, 0].max()), float(finite[:, 1].max())
    if not (np.isfinite([x0, y0, x1, y1]).all()) or (x1 - x0) < 1e-3 or (y1 - y0) < 1e-3:
        return None
    return x0, y0, x1, y1


def warp_segmentation_mask_topdown_bgr(
    bgr: np.ndarray,
    mask_hw: np.ndarray,
    src_quad_tl_tr_br_bl: np.ndarray,
    out_w: int,
    out_h: int,
) -> Optional[np.ndarray]:
    """
    Reproject the masked object to a top-down view and **auto-frame** it in the panel.

    The green quad defines a single homography ``H`` (image → plan; see ``compute_table_topdown_homography``)
    that is applied to the whole frame. The plan-space bounding box of the mask is then letterboxed (aspect
    preserved) into ``out_w``×``out_h`` via an affine ``S``, and the frame is sampled with ``M = S @ H`` so
    the object fills the panel wherever it sits relative to the calibration quad. RGB is kept only where the
    warped mask is positive. ``mask_hw`` must match ``bgr`` size.
    """
    if out_w < 2 or out_h < 2:
        return None
    if bgr.shape[:2] != mask_hw.shape[:2]:
        return None
    if not (mask_hw > 0).any():
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    H = compute_table_topdown_homography(
        src_quad_tl_tr_br_bl, image_w=bgr.shape[1], image_h=bgr.shape[0]
    )
    if H is None or not np.isfinite(H).all():
        return None

    bbox = _plan_bbox_of_mask(H, mask_hw)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    margin = 0.08
    x0 -= margin * bw
    x1 += margin * bw
    y0 -= margin * bh
    y1 += margin * bh
    bw = max(x1 - x0, 1e-3)
    bh = max(y1 - y0, 1e-3)

    ow = float(max(1, out_w - 1))
    oh = float(max(1, out_h - 1))
    scale = min(ow / bw, oh / bh)
    tx = 0.5 * (ow - scale * bw) - scale * x0
    ty = 0.5 * (oh - scale * bh) - scale * y0
    S = np.array([[scale, 0.0, tx], [0.0, scale, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    M = (S @ H.astype(np.float64)).astype(np.float32)
    if not np.isfinite(M).all():
        return None

    plan_bgr = cv2.warpPerspective(
        bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    m_u8 = (mask_hw > 0).astype(np.uint8) * 255
    plan_m_u8 = cv2.warpPerspective(
        m_u8,
        M,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    plan_m = plan_m_u8.astype(np.float32) / 255.0
    alpha = np.clip(plan_m[..., None], 0.0, 1.0)
    out = plan_bgr.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def topdown_panel_message_bgr(out_w: int, out_h: int, lines: List[str]) -> np.ndarray:
    """Dark panel with short status lines (fits narrow lower-right strip)."""
    panel = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    panel[:] = (28, 28, 28)
    y = 16
    for line in lines[:4]:
        if not line:
            continue
        short = line if len(line) <= 48 else line[:45] + "..."
        cv2.putText(
            panel,
            short,
            (4, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y += 18
    return panel


def draw_table_topdown_calibration_overlay(
    main_bgr: np.ndarray,
    clicks: List[Tuple[int, int]],
    src_quad: Optional[np.ndarray],
) -> None:
    """Draw homography helpers on the main BGR panel (in place)."""
    if src_quad is not None:
        q = np.asarray(src_quad, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(main_bgr, [q], True, (0, 200, 0), 2)
    if len(clicks) >= 2:
        pts = np.array(clicks, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(main_bgr, [pts], False, (0, 255, 255), 1)
    for i, (px, py) in enumerate(clicks):
        cv2.circle(main_bgr, (int(px), int(py)), 6, (0, 255, 255), 2)
        cv2.putText(
            main_bgr,
            str(i + 1),
            (int(px) + 8, int(py) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


class FrameSource(ABC):
    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return RGB uint8 HxWx3 or None."""

    @abstractmethod
    def close(self) -> None:
        pass


class WebcamSource(FrameSource):
    """USB / UVC webcam. Requests low buffer by default to reduce motion blur from stale queued frames."""

    def __init__(
        self,
        device: int | str = 0,
        width: int = 0,
        height: int = 0,
        fps: float = 0.0,
        buffer_size: int = 1,
        flush_grabs: int = 0,
    ):
        self.flush_grabs = max(0, int(flush_grabs))
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam {device!r}")
        if width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, float(fps))
        # Minimize internal queue where the backend honors it (e.g. many V4L2 builds).
        if buffer_size >= 0:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, float(buffer_size))

    def read(self) -> Optional[np.ndarray]:
        for _ in range(self.flush_grabs):
            if not self.cap.grab():
                break
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


def build_frame_source(
    source: str,
    path: Optional[str],
    webcam_id: int,
    *,
    webcam_device: Optional[str] = None,
    webcam_width: int = 0,
    webcam_height: int = 0,
    webcam_fps: float = 0.0,
    webcam_buffer_size: int = 1,
    webcam_flush_grabs: int = 0,
) -> FrameSource:
    if source == "webcam":
        dev: int | str = webcam_id
        if webcam_device is not None and webcam_device.strip():
            dev = webcam_device.strip()
        return WebcamSource(
            dev,
            width=webcam_width,
            height=webcam_height,
            fps=webcam_fps,
            buffer_size=webcam_buffer_size,
            flush_grabs=webcam_flush_grabs,
        )
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
    stream_cfg: StreamStabilityConfig = field(default_factory=StreamStabilityConfig)

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
    # Webcam / RealSense: SAM2VideoPredictor state built from an in-memory tensor clip (append per frame).
    stream_video_state: Optional[Dict[str, Any]] = None
    stream_video_state_cap: int = 360
    stream_video_offload_video: bool = False
    stream_video_offload_state: bool = False
    # If >0, log timing / mask stats every N SAM2 stream propagates (counts attempts, not only successes).
    stream_diag_every: int = 0
    stream_diag_prop_count: int = 0
    file_frame_index: int = 0
    last_bgr_display: Optional[np.ndarray] = None

    # Stream (webcam / realsense): frozen copy of the mask at lock — gates new predictions.
    anchor_mask: Optional[np.ndarray] = None
    # event_only + TRACKING: mask drawn until the next Space (tracking still updates locked_mask).
    event_only_overlay_mask: Optional[np.ndarray] = None
    _stream_stability_ticks: int = 0

    # After lock, skip one IoU check so we do not compare centroid refinement to the multimask.
    _skip_iou_once: bool = False
    # GUI: width of the main video column in the composite window (clicks to the right are ignored).
    gui_main_panel_width: int = 0
    # Optional table plane: four image clicks (TL,TR,BR,BL of plan rectangle) for lower-panel homography.
    table_cal_clicks: List[Tuple[int, int]] = field(default_factory=list)
    table_topdown_src_quad: Optional[np.ndarray] = None  # (4,2) float32 when calibration succeeded

    def __post_init__(self) -> None:
        self._maybe_autocast = nullcontext()
        if self.device.type == "cuda":
            self._maybe_autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    def reset_table_topdown_cal(self) -> None:
        """Clear interactive table-plane clicks and homography (``t`` key when ``--top-down on``)."""
        self.table_cal_clicks = []
        self.table_topdown_src_quad = None

    def begin_stream_click_repick(self, x: int, y: int) -> None:
        """Webcam/RealSense: new click during TRACKING — discard stream memory and pick a fresh point."""
        self.stream_video_state = None
        self.lock_state = LockState.NEED_CLICK
        self.pending_click_xy = (int(x), int(y))
        self.candidate_masks = None
        self.candidate_scores = None
        self.lock_prompt_pts = None
        self.lock_prompt_labels = None
        self.locked_mask = None
        self.prev_mask = None
        self.anchor_mask = None
        self.event_only_overlay_mask = None
        self._skip_iou_once = False
        self._stream_stability_ticks = 0
        self.stream_diag_prop_count = 0

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
        self.anchor_mask = None
        self.event_only_overlay_mask = None
        self._stream_stability_ticks = 0
        self.stream_video_state = None
        self.stream_diag_prop_count = 0

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

    def init_stream_video_from_lock(
        self,
        rgb: np.ndarray,
        points_xy: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """After multimask lock: one-frame video state + points on frame 0 + propagate (SAM2 memory seed)."""
        h, w = rgb.shape[:2]
        sz = int(self.video_predictor.image_size)
        offload_v = self.stream_video_offload_video
        batch = rgb_hwc_uint8_to_sam2_image_tensor(rgb, sz, self.device, offload_v).unsqueeze(0)
        with torch.inference_mode(), self._maybe_autocast:
            self.stream_video_state = init_video_state_from_images_tensor(
                self.video_predictor,
                batch,
                video_height=h,
                video_width=w,
                offload_video_to_cpu=offload_v,
                offload_state_to_cpu=self.stream_video_offload_state,
            )
            self.video_predictor.add_new_points_or_box(
                self.stream_video_state,
                frame_idx=0,
                obj_id=OBJ_ID,
                points=torch.as_tensor(points_xy, dtype=torch.float32),
                labels=torch.as_tensor(labels, dtype=torch.int32),
                clear_old_points=True,
                normalize_coords=True,
                box=None,
            )
            last_mask: Optional[np.ndarray] = None
            with _silence_sam2_video_tqdm():
                for _f_idx, _obj_ids, video_res_masks in self.video_predictor.propagate_in_video(
                    self.stream_video_state,
                    start_frame_idx=0,
                    max_frame_num_to_track=1,
                    reverse=False,
                ):
                    last_mask = video_res_masks[0, 0].detach().cpu().numpy() > 0.5
        if last_mask is not None:
            self.locked_mask = mask_to_shape_hw(last_mask.astype(bool), h, w)
            self.anchor_mask = self.locked_mask.copy()

    def _stream_video_reanchor(self, rgb: np.ndarray) -> None:
        """Re-init clip with current RGB + last mask (``add_new_mask``) when frame buffer hits cap."""
        if self.locked_mask is None:
            return
        h, w = rgb.shape[:2]
        mask = self.locked_mask
        if mask.shape != (h, w):
            mask = (cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5)
        sz = int(self.video_predictor.image_size)
        offload_v = self.stream_video_offload_video
        batch = rgb_hwc_uint8_to_sam2_image_tensor(rgb, sz, self.device, offload_v).unsqueeze(0)
        with torch.inference_mode(), self._maybe_autocast:
            self.stream_video_state = init_video_state_from_images_tensor(
                self.video_predictor,
                batch,
                video_height=h,
                video_width=w,
                offload_video_to_cpu=offload_v,
                offload_state_to_cpu=self.stream_video_offload_state,
            )
            self.video_predictor.add_new_mask(
                self.stream_video_state,
                frame_idx=0,
                obj_id=OBJ_ID,
                mask=torch.as_tensor(mask, dtype=torch.bool),
            )
            last_mask: Optional[np.ndarray] = None
            with _silence_sam2_video_tqdm():
                for _f_idx, _obj_ids, video_res_masks in self.video_predictor.propagate_in_video(
                    self.stream_video_state,
                    start_frame_idx=0,
                    max_frame_num_to_track=1,
                    reverse=False,
                ):
                    last_mask = video_res_masks[0, 0].detach().cpu().numpy() > 0.5
        if last_mask is not None:
            self.locked_mask = mask_to_shape_hw(last_mask.astype(bool), h, w)
        self.prev_mask = None

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

    def finalize_multimask_lock(
        self,
        rgb: np.ndarray,
        h: int,
        w: int,
        *,
        event_only: bool,
        source: str,
        video_disk_path: Optional[str],
        frame_source: FrameSource,
        progress_win: str,
    ) -> bool:
        """Apply the current multimask as ``locked_mask`` and start file/stream tracking."""
        self.locked_mask = self.current_chosen_mask()
        if self.locked_mask is None:
            return False
        self.prev_mask = None
        self.lock_state = LockState.TRACKING
        self.candidate_masks = None
        self.candidate_scores = None
        self.anchor_mask = self.locked_mask.astype(bool).copy()
        self._stream_stability_ticks = 0
        if event_only and source in ("webcam", "realsense"):
            lm = self.locked_mask.astype(bool)
            if lm.shape != (h, w):
                lm = (cv2.resize(lm.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5)
            self.event_only_overlay_mask = lm.astype(bool)
        else:
            self.event_only_overlay_mask = None
        self._skip_iou_once = True
        if source in ("webcam", "realsense"):
            if self.lock_prompt_pts is not None and self.lock_prompt_labels is not None:
                self.init_stream_video_from_lock(rgb, self.lock_prompt_pts, self.lock_prompt_labels)
        if video_disk_path is not None and self.lock_prompt_pts is not None:
            logging.info("Video propagate on %s …", video_disk_path)
            self.reset_video_file_state(video_disk_path)
            bgr_snap = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self.run_full_propagation_file(
                0,
                self.lock_prompt_pts,
                self.lock_prompt_labels,
                progress_win=progress_win,
                progress_bgr=bgr_snap,
            )
            self.file_frame_index = 0
            _sync_file_frame(frame_source, 0)
        logging.info("Mask locked; tracking.")
        return True

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

    def _dilated_anchor_mask(self, h: int, w: int) -> Optional[np.ndarray]:
        if self.anchor_mask is None:
            return None
        a = self.anchor_mask.astype(np.uint8)
        if a.shape != (h, w):
            a = (cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        k = max(3, int(round(self.stream_cfg.anchor_dilate_ratio * float(min(h, w)))))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(a, kernel) > 0

    def _stream_anchor_overlap_threshold(self) -> float:
        u = min(1.0, self._stream_stability_ticks / max(1, self.stream_cfg.anchor_overlap_decay_frames))
        s, e = self.stream_cfg.anchor_overlap_start, self.stream_cfg.anchor_overlap_end
        return float(s + (e - s) * u)

    def stream_accepts_candidate(self, new: np.ndarray, h: int, w: int) -> bool:
        """Reject centroid-SAM candidates that drift to other objects or teleport."""
        if new.shape != (h, w):
            new = (cv2.resize(new.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5)

        if not self.tracking_health_ok(new, iou_ref=None):
            return False

        c_new = _mask_centroid(new)
        if c_new is None:
            return False

        if self.locked_mask is not None and self.locked_mask.shape == (h, w):
            c_old = _mask_centroid(self.locked_mask)
            if c_old is not None:
                diag = float((h * h + w * w) ** 0.5)
                lim = self.stream_cfg.max_centroid_step_frac * diag
                if self._skip_iou_once:
                    lim *= self.stream_cfg.first_refine_centroid_step_mult
                if _centroid_distance(c_new, c_old) > lim:
                    return False

        if not self._skip_iou_once and self.locked_mask is not None and self.locked_mask.shape == new.shape:
            if _mask_iou(new, self.locked_mask) < self.stream_cfg.min_iou_vs_prev:
                return False

        if self.anchor_mask is not None:
            dil = self._dilated_anchor_mask(h, w)
            if dil is not None:
                n_pix = int(new.sum())
                if n_pix < 1:
                    return False
                overlap = float(np.logical_and(new, dil).sum()) / float(n_pix)
                if overlap < self._stream_anchor_overlap_threshold():
                    return False

        return True

    def _stream_diag_log_propagate(
        self,
        st: Dict[str, Any],
        prop_ms: float,
        last_mask: Optional[np.ndarray],
        h: int,
        w: int,
        prev_locked: Optional[np.ndarray],
        err: Optional[str],
    ) -> None:
        """Periodic INFO log for webcam stream propagates (attempt-based, not success-only)."""
        if self.stream_diag_every <= 0:
            return
        self.stream_diag_prop_count += 1
        if self.stream_diag_prop_count % self.stream_diag_every != 0:
            return
        buf = int(st.get("num_frames", -1))
        if err is not None:
            logging.info(
                "stream_diag attempt=%s buf_frames=%s prop_ms=%.1f status=ERROR %s",
                self.stream_diag_prop_count,
                buf,
                prop_ms,
                err,
            )
            return
        has_m = last_mask is not None and bool(last_mask.any())
        if has_m and last_mask is not None:
            tmp = mask_to_shape_hw(last_mask.astype(bool), h, w)
            mf = float(tmp.mean())
            iou = (
                _mask_iou(tmp, prev_locked)
                if prev_locked is not None and prev_locked.shape == tmp.shape
                else -1.0
            )
        else:
            mf = 0.0
            iou = -1.0
        logging.info(
            "stream_diag attempt=%s buf_frames=%s prop_ms=%.1f has_mask=%d mask_frac=%.4f iou_vs_prev=%.4f",
            self.stream_diag_prop_count,
            buf,
            prop_ms,
            int(has_m),
            mf,
            iou,
        )

    def on_tracking_tick_stream(self, rgb: np.ndarray) -> np.ndarray:
        """Append the new frame to the in-memory clip and run ``propagate_in_video`` for that index only."""
        h, w = rgb.shape[:2]
        if self.locked_mask is None:
            return np.zeros((h, w), dtype=bool)
        if self.stream_video_state is None:
            logging.warning("stream SAM2 video: missing stream_video_state; holding last mask")
            return self.locked_mask.copy()
        if self.locked_mask.shape != (h, w):
            self.locked_mask = (
                cv2.resize(self.locked_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
            )
        if self.anchor_mask is not None and self.anchor_mask.shape != (h, w):
            self.anchor_mask = (
                cv2.resize(self.anchor_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
            )

        cap = self.stream_video_state_cap
        if cap > 0 and int(self.stream_video_state["num_frames"]) >= cap:
            self._stream_video_reanchor(rgb)
            return self.locked_mask.copy()

        sz = int(self.video_predictor.image_size)
        offload_v = self.stream_video_offload_video
        new_fr = rgb_hwc_uint8_to_sam2_image_tensor(rgb, sz, self.device, offload_v).unsqueeze(0)
        st = self.stream_video_state
        ref = st["images"]
        new_fr = new_fr.to(device=ref.device, dtype=ref.dtype)
        t_prop0 = time.perf_counter()
        try:
            with torch.inference_mode(), self._maybe_autocast:
                st["images"] = torch.cat([ref, new_fr], dim=0)
                st["num_frames"] = int(st["images"].shape[0])
                t_new = st["num_frames"] - 1
                last_mask: Optional[np.ndarray] = None
                with _silence_sam2_video_tqdm():
                    for _f_idx, _obj_ids, video_res_masks in self.video_predictor.propagate_in_video(
                        st,
                        start_frame_idx=t_new,
                        max_frame_num_to_track=1,
                        reverse=False,
                    ):
                        last_mask = video_res_masks[0, 0].detach().cpu().numpy() > 0.5
        except Exception as e:
            prop_ms = (time.perf_counter() - t_prop0) * 1000.0
            self._stream_diag_log_propagate(st, prop_ms, None, h, w, self.locked_mask, repr(e))
            logging.exception("stream SAM2 video: propagate failed; holding previous mask")
            return self.locked_mask.copy()
        prop_ms = (time.perf_counter() - t_prop0) * 1000.0
        prev_locked = self.locked_mask.copy() if self.locked_mask is not None else None
        self._stream_diag_log_propagate(st, prop_ms, last_mask, h, w, prev_locked, None)

        if last_mask is None or not last_mask.any():
            logging.debug("stream SAM2 video: empty mask; holding previous mask")
            return self.locked_mask.copy()

        self._skip_iou_once = False
        self.prev_mask = prev_locked
        new_locked = mask_to_shape_hw(last_mask.astype(bool), h, w)
        self.locked_mask = new_locked
        return self.locked_mask.copy()

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
    """Return raw key code from OpenCV (waitKeyEx when available for extended codes)."""
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
    ap.add_argument("--webcam-id", type=int, default=0, help="OpenCV capture index (default 0). Ignored if --webcam-device is set.")
    ap.add_argument(
        "--webcam-device",
        type=str,
        default=None,
        help="V4L2 device path, e.g. /dev/video2. Overrides --webcam-id. Only with --source webcam.",
    )
    ap.add_argument("--webcam-fps", type=float, default=0.0, help="Optional CAP_PROP_FPS (0 = driver default).")
    ap.add_argument("--webcam-width", type=int, default=0, help="Optional capture width (0 = driver default).")
    ap.add_argument("--webcam-height", type=int, default=0, help="Optional capture height (0 = driver default).")
    ap.add_argument(
        "--webcam-buffer",
        type=int,
        default=1,
        help="CAP_PROP_BUFFERSIZE where supported; 1 reduces stale-frame latency (-1 = do not set).",
    )
    ap.add_argument(
        "--webcam-flush-grabs",
        type=int,
        default=0,
        help="grab() this many times before each read to drop queued frames (try 1–3 if motion looks delayed).",
    )
    ap.add_argument(
        "--stream-motion",
        choices=("default", "fast"),
        default="default",
        help="fast: looser IoU/anchor/centroid gates (legacy; only used if centroid stream helpers are re-enabled).",
    )
    ap.add_argument(
        "--stream-video-state-cap",
        type=int,
        default=360,
        help="Webcam/RealSense: max frames in the in-memory SAM2 clip before re-anchoring with add_new_mask "
        "(0 = unlimited; long runs may OOM as the frame tensor grows).",
    )
    ap.add_argument(
        "--stream-diag-every",
        type=int,
        default=0,
        help="Webcam/RealSense: if >0, log every N stream SAM2 propagate **attempts** (includes empty masks "
        "and errors; use 5–30). Logs only while TRACKING.",
    )
    ap.add_argument(
        "--lock-on",
        choices=("click", "enter"),
        default="click",
        help="click: after a point click, lock the top multimask and start tracking immediately (default). "
        "enter: click sets the point only; use , . or arrow keys to cycle masks, then Enter to lock.",
    )
    ap.add_argument(
        "--preview",
        choices=["live", "event_only", "auto"],
        default="live",
        help="live: overlay every frame. event_only: for webcam/realsense in TRACKING, overlay updates "
        "only on Space (tracking still runs each frame). Video/images always use a live overlay. "
        "auto: same as live.",
    )
    ap.add_argument(
        "--bb-orientation",
        choices=("aligned", "free"),
        default="aligned",
        help="Top-right panel when --top-down off (or before quad is set): aligned (default) = axis-aligned "
        "boundingRect; free = minAreaRect warp upright. When --top-down on and the quad is set, the upper "
        "panel is always the minAreaRect mask warp (table plan is only the lower panel).",
    )
    ap.add_argument(
        "--top-down",
        choices=("off", "on"),
        default="off",
        help="off: right column lower half stays black. on: four clicks define the table quad (the homography "
        "template); the lower half shows the masked object reprojected top-down and auto-framed. t=reset.",
    )
    ap.add_argument("--checkpoint", default="sam2_repo/checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    args = ap.parse_args()
    if args.preview == "auto":
        args.preview = "live"
    if args.webcam_device is not None and args.source != "webcam":
        ap.error("--webcam-device is only valid with --source webcam")
    if args.webcam_device and args.webcam_device.strip():
        wd = args.webcam_device.strip()
        if wd.startswith("/dev/") and not os.path.exists(wd):
            logging.warning("webcam path does not exist (yet): %s", wd)

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

    frame_source = build_frame_source(
        args.source,
        args.path,
        args.webcam_id,
        webcam_device=args.webcam_device,
        webcam_width=args.webcam_width,
        webcam_height=args.webcam_height,
        webcam_fps=args.webcam_fps,
        webcam_buffer_size=args.webcam_buffer,
        webcam_flush_grabs=args.webcam_flush_grabs,
    )

    session = InteractiveSession(
        device=device,
        image_predictor=image_predictor,
        video_predictor=video_predictor,
        frame_source=frame_source,
        source_kind=args.source,
        stream_video_state_cap=args.stream_video_state_cap,
        stream_diag_every=args.stream_diag_every,
    )
    if args.stream_diag_every > 0 and args.source in ("webcam", "realsense"):
        logging.info(
            "stream_diag: INFO every %s SAM2 propagate attempts while TRACKING. "
            "has_mask=0 means an empty prediction that tick.",
            args.stream_diag_every,
        )
    if args.stream_motion == "fast":
        session.stream_cfg = StreamStabilityConfig(
            min_iou_vs_prev=0.09,
            anchor_overlap_start=0.16,
            anchor_overlap_end=0.05,
            anchor_overlap_decay_frames=320,
            anchor_dilate_ratio=0.14,
            max_centroid_step_frac=0.105,
            first_refine_centroid_step_mult=2.5,
        )
        logging.info("stream-motion=fast: relaxed stream stability gates")

    if args.top_down == "on":
        logging.info(
            "Top-down mode: four table clicks (TL,TR,BR,BL) define the homography template. t resets."
        )

    video_disk_path: Optional[str] = None
    if args.source in ("video", "images"):
        if not args.path:
            ap.error("--path required for video/images")
        video_disk_path = os.path.abspath(args.path)

    is_file_source = args.source in ("video", "images")

    def mask_disp_after_new_lock(rgb_: np.ndarray, h_: int, w_: int) -> np.ndarray:
        if args.source in ("webcam", "realsense"):
            return session.on_tracking_tick_stream(rgb_)
        if is_file_source and session.video_masks_by_frame is not None:
            mf = session.on_tracking_tick_file()
            if mf is not None:
                if mf.shape != (h_, w_):
                    mf = cv2.resize(mf.astype(np.uint8), (w_, h_), interpolation=cv2.INTER_NEAREST) > 0
                return mf.astype(bool)
        return np.zeros((h_, w_), dtype=bool)

    win = "SAM2 interactive"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if session.gui_main_panel_width > 0 and x >= session.gui_main_panel_width:
            return
        if (
            args.top_down == "on"
            and session.table_topdown_src_quad is None
            and len(session.table_cal_clicks) < 4
        ):
            session.table_cal_clicks.append((int(x), int(y)))
            n = len(session.table_cal_clicks)
            logging.info("Table top-down: point %s/4 at (%s, %s)", n, x, y)
            if n == 4:
                quad = np.array(session.table_cal_clicks, dtype=np.float32)
                if abs(_quad_double_area(quad)) < 8.0:
                    logging.warning("Table quad nearly degenerate; removing last point.")
                    session.table_cal_clicks.pop()
                else:
                    probe_dst = np.array([[0, 0], [99, 0], [99, 99], [0, 99]], dtype=np.float32)
                    try:
                        _ = cv2.getPerspectiveTransform(quad, probe_dst)
                    except Exception:
                        logging.warning(
                            "Table homography failed (wrong click order?). Removing last point — "
                            "use plan TL, TR, BR, BL on the image."
                        )
                        session.table_cal_clicks.pop()
                    else:
                        session.table_topdown_src_quad = quad
                        session.table_cal_clicks.clear()
                        logging.info("Table top-down homography ready.")
            return
        if session.lock_state in (LockState.NEED_CLICK, LockState.CHOOSE_MASK):
            session.pending_click_xy = (x, y)
        elif session.lock_state == LockState.TRACKING and session.source_kind in ("webcam", "realsense"):
            session.begin_stream_click_repick(x, y)
            logging.info(
                "Stream: click repick — new multimask; %s",
                "locking top mask" if args.lock_on == "click" else "press Enter to lock (, . to cycle)",
            )

    cv2.setMouseCallback(win, on_mouse)

    _loop_last = time.perf_counter()
    _loop_fps_smooth = 0.0

    try:
        while True:
            _t_iter = time.perf_counter()
            _dt_iter = _t_iter - _loop_last
            _loop_last = _t_iter
            if _dt_iter > 1e-6:
                inst_hz = 1.0 / _dt_iter
                if _dt_iter < 2.0:
                    _loop_fps_smooth = (
                        0.88 * _loop_fps_smooth + 0.12 * inst_hz if _loop_fps_smooth > 0 else inst_hz
                    )

            is_file = is_file_source

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
            hz_note = f" ~{_loop_fps_smooth:.0f} Hz" if args.source in ("webcam", "realsense") else ""
            lock_help = (
                "click=lock+track (or stream re-pick) | Space=plan r=reprompt q=quit"
                if args.lock_on == "click"
                else "click=point , ./arrows=cycle Enter=lock (stream re-pick same) | Space=plan r=reprompt q=quit"
            )
            status = [
                f"{args.source} lock={session.lock_state.name} preview={args.preview} lock-on={args.lock_on}{hz_note}",
                lock_help,
            ]
            if args.top_down == "on":
                if session.table_topdown_src_quad is None:
                    status.append(
                        "top-down: click 4 coplanar corners — (1) plan upper-left (2) upper-right "
                        "(3) lower-right (4) lower-left of one table rectangle | t=reset"
                    )
                else:
                    status.append(
                        "top-down: homography active — upper=oriented mask rect | lower=reprojected mask "
                        "(auto-framed) | t=reset"
                    )
            event_only = args.preview == "event_only" and args.source in ("webcam", "realsense")
            # Avoid a stale frozen overlay if the user is not in event_only mode.
            if args.source in ("webcam", "realsense") and not event_only:
                session.event_only_overlay_mask = None

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
                multimask_from_new_click = False
                if session.pending_click_xy is not None:
                    x, y = session.pending_click_xy
                    pts = np.array([[x, y]], dtype=np.float64)
                    lbl = np.array([1], dtype=np.int64)
                    session.lock_prompt_pts = pts.copy()
                    session.lock_prompt_labels = lbl.copy()
                    session.image_reprompt_multimask(rgb, pts, lbl)
                    session.pending_click_xy = None
                    session.lock_state = LockState.CHOOSE_MASK
                    multimask_from_new_click = True
                if session.lock_state == LockState.CHOOSE_MASK and session.candidate_masks is not None:
                    cm = session.current_chosen_mask()
                    if cm is not None:
                        mask_disp = cm
                if (
                    args.lock_on == "click"
                    and multimask_from_new_click
                    and session.lock_state == LockState.CHOOSE_MASK
                    and session.candidate_masks is not None
                    and session.lock_prompt_pts is not None
                ):
                    if session.finalize_multimask_lock(
                        rgb,
                        h,
                        w,
                        event_only=event_only,
                        source=args.source,
                        video_disk_path=video_disk_path,
                        frame_source=frame_source,
                        progress_win=win,
                    ):
                        mask_disp = mask_disp_after_new_lock(rgb, h, w)

            if event_only and session.lock_state == LockState.TRACKING:
                if session.event_only_overlay_mask is None and session.locked_mask is not None:
                    session.event_only_overlay_mask = mask_to_shape_hw(session.locked_mask, h, w)
                if session.event_only_overlay_mask is not None:
                    om = mask_to_shape_hw(session.event_only_overlay_mask, h, w)
                    session.event_only_overlay_mask = om
                    mask_draw = om
                else:
                    mask_draw = mask_to_shape_hw(mask_disp, h, w)
                status.append("event_only: Space refreshes overlay")
            else:
                mask_draw = mask_to_shape_hw(mask_disp, h, w)

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            overlay = overlay_mask_bgr(bgr, mask_draw)
            main_panel = composite_status(overlay, status)
            if args.top_down == "on":
                draw_table_topdown_calibration_overlay(
                    main_panel, session.table_cal_clicks, session.table_topdown_src_quad
                )
            session.gui_main_panel_width = w
            strip_w = max(1, int(round(float(w) * 0.5)))
            ph_top = h // 2
            ph_bot = h - ph_top
            lower_panel_bgr: Optional[np.ndarray] = None
            upper_panel_bgr: Optional[np.ndarray] = None
            if args.top_down == "on" and session.table_topdown_src_quad is not None:
                q = session.table_topdown_src_quad
                if not (mask_draw > 0).any():
                    lower_panel_bgr = topdown_panel_message_bgr(
                        strip_w, ph_bot, ["No segmentation yet.", "Lock/track an object on the table."]
                    )
                else:
                    # Upper: tight oriented-rect unwarp of the mask (image-plane); distinct from table plan.
                    upper_panel_bgr = oriented_min_area_rect_mask_patch_bgr(
                        bgr, mask_draw, strip_w, ph_top
                    )
                    lower_panel_bgr = warp_segmentation_mask_topdown_bgr(
                        bgr,
                        mask_draw,
                        q,
                        strip_w,
                        ph_bot,
                    )
                    if lower_panel_bgr is None or int(lower_panel_bgr.sum()) == 0:
                        lower_panel_bgr = topdown_panel_message_bgr(
                            strip_w,
                            ph_bot,
                            ["Warp produced empty panel.", "Try re-ordering TL,TR,BR,BL (t)."],
                        )
            vis = composite_main_and_side_panels_bgr(
                main_panel,
                bgr,
                mask_draw,
                bb_orientation=args.bb_orientation,
                lower_panel_bgr=lower_panel_bgr,
                upper_panel_bgr=upper_panel_bgr,
            )
            session.last_bgr_display = vis

            cv2.imshow(win, vis)

            k = _poll_key()
            key = k & 0xFF

            if key in (27, ord("q")):
                break
            if key == ord("t") and args.top_down == "on":
                session.reset_table_topdown_cal()
                logging.info("Table top-down calibration cleared.")
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
                    pass
                elif session.finalize_multimask_lock(
                    rgb,
                    h,
                    w,
                    event_only=event_only,
                    source=args.source,
                    video_disk_path=video_disk_path,
                    frame_source=frame_source,
                    progress_win=win,
                ):
                    mask_disp = mask_disp_after_new_lock(rgb, h, w)

            if key == ord(" "):
                if event_only and session.lock_state == LockState.TRACKING:
                    src = session.locked_mask
                    if src is not None and src.any():
                        sm = src.astype(bool)
                        if sm.shape != (h, w):
                            sm = (
                                cv2.resize(sm.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
                            ).astype(bool)
                        session.event_only_overlay_mask = sm.copy()
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
