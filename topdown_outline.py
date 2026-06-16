"""
Classical top-down mask outline: denoise, largest connected component, external contour, 2D normals.

Designed for warped panel masks from ``warp_mask_topdown_u8`` / ``warp_segmentation_mask_topdown_bgr_and_mask``.
Optional Gaussian blur on the mask boundary and **corner-aware** contour smoothing straighten long edges
while keeping corners comparatively tight (avoid global circular smoothing that rounds every vertex).

Image-plane fusion feeds the warp: past masks and the anchor can fill **brief, local** occlusions.
By default, past/anchor pixels are intersected with a **dilated current mask** so a moved object does not
leave a permanent ghost at its previous pose (unclipped temporal OR). A **connected-component filter**
then keeps only regions that touch the raw current mask, and outline history can be **pruned** by IoU
and centroid distance before fusion.

A future CNN can consume the same panel frame stacked with normalized (x, y) coordinates per the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class OutlineResult:
    """Denoised mask and largest external contour in panel pixel coordinates."""

    mask_clean_u8: np.ndarray  # (H, W) uint8 0 / 255
    contour_xy: np.ndarray  # (N, 2) float32, column order (x, y)
    normals_xy: np.ndarray  # (N, 2) float32, unit vectors (outward-consistent with OpenCV outer contour)


def _odd_kernel_size(kernel_px: int) -> int:
    k = int(kernel_px)
    if k < 1:
        return 1
    if k % 2 == 0:
        k += 1
    return k


def _largest_component_mask(bin_u8: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return binary mask (uint8 0/255) of the largest 8-connected foreground component and its area."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats((bin_u8 > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return np.zeros_like(bin_u8), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(1 + int(np.argmax(areas)))
    out = np.zeros_like(bin_u8)
    out[labels == best] = 255
    return out, int(areas.max())


def _bool_mask_hw(m: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize mask to (h, w) and return bool."""
    m = np.asarray(m)
    if m.ndim != 2:
        raise ValueError("mask must be 2D (H, W)")
    if m.shape[0] == h and m.shape[1] == w:
        return (m > 0).astype(bool)
    resized = cv2.resize((m > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(bool)


def keep_fused_components_touching_reference(fused_bool: np.ndarray, ref_bool: np.ndarray) -> np.ndarray:
    """
    Union of 8-connected components of ``fused_bool`` that have positive overlap with ``ref_bool``.

    If nothing overlaps ``ref``, returns ``ref_bool`` only (avoids showing a pure ghost). If ``fused_bool``
    is empty, returns an empty mask.
    """
    fused = np.asarray(fused_bool, dtype=bool)
    ref = np.asarray(ref_bool, dtype=bool)
    if fused.shape != ref.shape:
        raise ValueError("fused_bool and ref_bool must match shape")
    if not fused.any():
        return np.zeros_like(fused, dtype=bool)
    if not ref.any():
        return fused.copy()
    fg = (fused.astype(np.uint8) * 255)
    num, labels, _, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    keep = np.zeros_like(fused, dtype=bool)
    for lid in range(1, int(num)):
        m = labels == lid
        if np.any(m & ref):
            keep |= m
    if not keep.any():
        return ref.copy()
    return keep


def prune_outline_history_masks(
    past_seq: Sequence[np.ndarray],
    current_hw: np.ndarray,
    h: int,
    w: int,
    *,
    iou_min: float,
    max_centroid_px: float,
) -> List[np.ndarray]:
    """
    Drop past masks that are spatially inconsistent with ``current_hw``.

    Keeps a past mask if ``IoU(past, current) >= iou_min`` **or** centroid distance is at most
    ``max_centroid_px`` (when both centroids exist).
    """
    cur = _bool_mask_hw(current_hw, h, w)
    if not cur.any():
        return list(past_seq)
    d_max = float(max_centroid_px)
    if d_max <= 0.0:
        d_max = max(48.0, 0.25 * float(min(h, w)))
    ys, xs = np.where(cur)
    cur_c = (float(xs.mean()), float(ys.mean()))
    out: List[np.ndarray] = []
    for p in past_seq:
        pm = _bool_mask_hw(p, h, w)
        if not pm.any():
            continue
        inter = int(np.logical_and(pm, cur).sum())
        union = int(np.logical_or(pm, cur).sum())
        iou = float(inter / union) if union > 0 else 0.0
        ok_iou = iou >= float(iou_min)
        py, px = np.where(pm)
        pc = (float(px.mean()), float(py.mean()))
        dist = float(np.hypot(pc[0] - cur_c[0], pc[1] - cur_c[1]))
        ok_cent = dist <= d_max
        if ok_iou or ok_cent:
            out.append(np.asarray(p))
    return out


def _fusion_dilate_radius(out_h: int, out_w: int, fusion_dilate_radius_px: int) -> int:
    rad = int(fusion_dilate_radius_px)
    if rad <= 0:
        rad = max(20, min(out_h, out_w) // 10)
    return rad


def fuse_outline_masks_image_plane(
    current_hw: np.ndarray,
    past_hw_seq: Sequence[np.ndarray],
    anchor_hw: Optional[np.ndarray],
    *,
    out_h: int,
    out_w: int,
    fuse_anchor: bool,
    clip_past_and_anchor_to_dilated_current: bool = True,
    fusion_dilate_radius_px: int = 0,
    cc_filter_touching_current: bool = True,
) -> np.ndarray:
    """
    Pixelwise fusion of binary masks in the image plane, then ``uint8`` 0/255 of shape ``(out_h, out_w)``.

    Use **before** top-down warp. ``past_hw_seq`` should hold prior frames only (not the current frame).

    When ``clip_past_and_anchor_to_dilated_current`` is True and the current mask is non-empty, each
    past mask and the anchor are combined as ``past & dilate(current)`` (then OR'd with ``current``).
    That keeps temporal/anchor support **near the live segmentation** so moved objects do not retain a
    second blob at an old location.

    When ``cc_filter_touching_current`` is True and the current mask is non-empty, after fusion only
    connected components that overlap the **raw** current mask are kept (removes ghosts that still lie
    inside ``dilate(current)`` but not on live pixels).

    When the current mask is empty: ``past[-1]`` OR ``(anchor & dilate(past[-1]))`` if past exists; else
    anchor only — anchor cannot add a blob far from the last visible mask.

    Set ``clip_past_and_anchor_to_dilated_current=False`` to restore legacy global OR (not recommended
    if the object can translate in the image).
    """
    cur = _bool_mask_hw(current_hw, out_h, out_w)

    def _maybe_cc(fused_bool: np.ndarray) -> np.ndarray:
        if cc_filter_touching_current and cur.any():
            return keep_fused_components_touching_reference(fused_bool, cur)
        return fused_bool

    if clip_past_and_anchor_to_dilated_current:
        if cur.any():
            rad = _fusion_dilate_radius(out_h, out_w, fusion_dilate_radius_px)
            k = max(3, 2 * rad + 1)
            el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            near_u8 = cv2.dilate((cur.astype(np.uint8) * 255), el, iterations=1)
            near = near_u8 > 0
            fused = cur.copy()
            for p in past_hw_seq:
                fused |= _bool_mask_hw(p, out_h, out_w) & near
            if fuse_anchor and anchor_hw is not None:
                fused |= _bool_mask_hw(anchor_hw, out_h, out_w) & near
            fused = _maybe_cc(fused)
            return (fused.astype(np.uint8) * 255)
        # No current foreground: do not OR the whole deque (prevents stale multi-blob ghosts).
        if past_hw_seq:
            last = _bool_mask_hw(past_hw_seq[-1], out_h, out_w)
            fused = last.copy()
            if fuse_anchor and anchor_hw is not None:
                rad = _fusion_dilate_radius(out_h, out_w, fusion_dilate_radius_px)
                kk = max(3, 2 * rad + 1)
                el2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
                dil_last = cv2.dilate((last.astype(np.uint8) * 255), el2, iterations=1) > 0
                fused |= _bool_mask_hw(anchor_hw, out_h, out_w) & dil_last
        elif fuse_anchor and anchor_hw is not None:
            fused = _bool_mask_hw(anchor_hw, out_h, out_w)
        else:
            fused = cur.copy()
        return (fused.astype(np.uint8) * 255)

    fused = cur.copy()
    for p in past_hw_seq:
        fused |= _bool_mask_hw(p, out_h, out_w)
    if fuse_anchor and anchor_hw is not None:
        fused |= _bool_mask_hw(anchor_hw, out_h, out_w)
    if cur.any():
        fused = _maybe_cc(fused)
    return (fused.astype(np.uint8) * 255)


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 1e-9:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    k /= k.sum()
    return k


def _circular_convolve_1d(v: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    n = int(v.shape[0])
    r = int(kernel.shape[0] // 2)
    vpad = np.concatenate([v[-r:], v, v[:r]])
    out = np.convolve(vpad, kernel, mode="valid")
    if out.shape[0] != n:
        return v.astype(np.float64)
    return out


def _open_gaussian_smooth_chain_fixed_ends(seg: np.ndarray, sigma_pts: float) -> np.ndarray:
    """1D Gaussian along vertex index; endpoints fixed (open chain). ``seg`` is (L, 2)."""
    if seg.shape[0] < 3 or sigma_pts <= 1e-9:
        return np.asarray(seg, dtype=np.float64)
    kernel = _gaussian_kernel_1d(float(sigma_pts))
    r = int(kernel.shape[0] // 2)
    out = np.zeros_like(seg, dtype=np.float64)
    for d in range(2):
        v = seg[:, d].astype(np.float64)
        vpad = np.concatenate([[v[0]] * r, v, [v[-1]] * r])
        sm = np.convolve(vpad, kernel, mode="valid")
        if sm.shape[0] != seg.shape[0]:
            return seg.astype(np.float64)
        out[:, d] = sm
    out[0] = seg[0]
    out[-1] = seg[-1]
    return out


def _max_point_to_segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Max Euclidean distance from each row of ``pts`` to closed segment ``ab``."""
    ab = b.astype(np.float64) - a.astype(np.float64)
    lab2 = float(np.dot(ab, ab)) + 1e-12
    ap = pts.astype(np.float64) - a.astype(np.float64)
    t = (ap @ ab) / lab2
    t = np.clip(t, 0.0, 1.0)
    proj = a.astype(np.float64) + (t[:, None] * ab)
    d = np.linalg.norm(pts.astype(np.float64) - proj, axis=1)
    return float(np.max(d))


def _resample_chord(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return np.stack([a, b], axis=0)[: max(n, 1)]
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
    return (a.astype(np.float64) * (1.0 - t) + b.astype(np.float64) * t)


def _vertex_turn_angle_rad(xy: np.ndarray) -> np.ndarray:
    """
    For each vertex ``i``, angle in ``[0, pi]`` between edge ``(i-1)->i`` and ``i->(i+1)``.
    Near **0** = straight; near **pi** = sharp reversal (rare on CCW outer contour).
    """
    c = np.asarray(xy, dtype=np.float64)
    n = c.shape[0]
    u = c - np.roll(c, 1, axis=0)  # p[i] - p[i-1]
    w = np.roll(c, -1, axis=0) - c  # p[i+1] - p[i]
    nu = np.linalg.norm(u, axis=1)
    nw = np.linalg.norm(w, axis=1)
    cosang = np.sum(u * w, axis=1) / (nu * nw + 1e-12)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.arccos(cosang)


def smooth_contour_straight_segments_preserve_corners(
    xy: np.ndarray,
    sigma_pts: float,
    *,
    corner_angle_deg: float = 14.0,
    edge_straighten_max_dev_px: float = 2.8,
) -> np.ndarray:
    """
    Smooth a **closed** polyline: nearly straight runs are collapsed toward their chord; curved runs get
    open-chain Gaussian smoothing with **fixed endpoints** (corners), avoiding global circular blur that
    rounds every corner uniformly.
    """
    c = np.asarray(xy, dtype=np.float64)
    n = c.shape[0]
    if n < 4 or sigma_pts <= 1e-9:
        return c.astype(np.float32)

    thr = np.deg2rad(float(corner_angle_deg))
    ang = _vertex_turn_angle_rad(c)
    is_corner = ang > thr
    corner_idx = np.flatnonzero(is_corner)

    if corner_idx.size == 0:
        return smooth_closed_polyline_xy(c.astype(np.float32), float(sigma_pts) * 0.35)

    if corner_idx.size == 1:
        return smooth_closed_polyline_xy(c.astype(np.float32), float(sigma_pts) * 0.45)

    m = int(corner_idx.size)
    pieces: List[np.ndarray] = []
    for k in range(m):
        i0 = int(corner_idx[k])
        i1 = int(corner_idx[(k + 1) % m])
        if i1 >= i0:
            inds = np.arange(i0, i1 + 1, dtype=int)
        else:
            inds = np.concatenate([np.arange(i0, n, dtype=int), np.arange(0, i1 + 1, dtype=int)])
        seg = c[inds]
        L = int(seg.shape[0])
        if L < 2:
            continue
        a, b = seg[0], seg[-1]
        if L == 2:
            sm = seg
        elif _max_point_to_segment_distance(seg, a, b) <= float(edge_straighten_max_dev_px):
            sm = _resample_chord(a, b, L)
        else:
            sm = _open_gaussian_smooth_chain_fixed_ends(seg, float(sigma_pts))
        if pieces:
            sm = sm[1:]  # drop duplicate joint
        pieces.append(sm)
    out = np.vstack(pieces) if pieces else c
    if out.shape[0] < 3:
        return c.astype(np.float32)
    return out.astype(np.float32)


def smooth_closed_polyline_xy(xy: np.ndarray, sigma_pts: float) -> np.ndarray:
    """Circular Gaussian smoothing along vertex index (closed curve)."""
    if sigma_pts <= 1e-9:
        return np.asarray(xy, dtype=np.float32)
    c = np.asarray(xy, dtype=np.float64)
    n = c.shape[0]
    if n < 3:
        return c.astype(np.float32)
    kernel = _gaussian_kernel_1d(float(sigma_pts))
    if kernel.size > n:
        kernel = _gaussian_kernel_1d(float(sigma_pts) * (n / max(kernel.size, 1)))
    sx = _circular_convolve_1d(c[:, 0], kernel)
    sy = _circular_convolve_1d(c[:, 1], kernel)
    return np.stack([sx, sy], axis=1).astype(np.float32)


def _contour_normals_unit(contour_xy: np.ndarray) -> np.ndarray:
    """Unit normals from central differences on a closed polyline (CCW tangent rotated -90° in image coords)."""
    c = np.asarray(contour_xy, dtype=np.float64)
    n = c.shape[0]
    if n < 3:
        return np.zeros((n, 2), dtype=np.float32)
    prev_idx = np.arange(n) - 1
    prev_idx[0] = n - 1
    next_idx = np.arange(n) + 1
    next_idx[-1] = 0
    tang = c[next_idx] - c[prev_idx]
    lengths = np.linalg.norm(tang, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-9)
    tang_u = tang / lengths
    nx = -tang_u[:, 1]
    ny = tang_u[:, 0]
    nl = np.linalg.norm(np.stack([nx, ny], axis=1), axis=1, keepdims=True)
    nl = np.maximum(nl, 1e-9)
    out = np.stack([nx, ny], axis=1) / nl
    return out.astype(np.float32)


def outline_from_topdown_mask(
    plan_mask_u8: np.ndarray,
    *,
    kernel_px: int = 3,
    min_area_frac: float = 0.0,
    epsilon_poly: float = 0.0,
    boundary_blur_sigma: float = 0.0,
    contour_smooth_sigma_pts: float = 0.0,
    contour_preserve_straight_edges: bool = True,
    contour_corner_angle_deg: float = 18.0,
    edge_straighten_max_dev_px: float = 2.8,
) -> OutlineResult:
    """
    Denoise ``plan_mask_u8`` (0 / 255), keep largest foreground blob, extract largest external contour.

    Parameters
    ----------
    plan_mask_u8
        Top-down warped binary mask.
    kernel_px
        Odd morphological kernel size (ellipse). If ``< 2``, morphology is skipped.
    min_area_frac
        If ``> 0``, require largest CC area ≥ this fraction of the **foreground pixel count** in the
        morphed mask before CC filtering; otherwise return an empty contour and zero mask.
    epsilon_poly
        If ``> 0``, ``cv2.approxPolyDP`` epsilon for simplifying the contour after extraction (before
        vertex smoothing when both are set).
    boundary_blur_sigma
        If ``> 0``, Gaussian-blur the largest-component mask, threshold at 0.5, take largest CC again,
        then extract contour (smoother boundary than raw segmentation).
    contour_smooth_sigma_pts
        If ``> 0``, smooth contour vertices. With ``contour_preserve_straight_edges`` (default), long
        nearly collinear runs are snapped to a chord and curved arcs between corners use open Gaussian
        smoothing (endpoints fixed) instead of global circular smoothing that rounds every corner.
    contour_preserve_straight_edges
        When True (default) and ``contour_smooth_sigma_pts > 0``, use corner-aware segment smoothing.
    contour_corner_angle_deg
        Treat a vertex as a corner if the turn angle (radially) exceeds this (degrees).
    edge_straighten_max_dev_px
        If all points on a segment deviate at most this from the chord, replace the segment by the chord.
    """
    if plan_mask_u8.ndim != 2:
        raise ValueError("plan_mask_u8 must be 2D (H, W)")
    h, w = plan_mask_u8.shape[:2]
    empty = OutlineResult(
        mask_clean_u8=np.zeros((h, w), dtype=np.uint8),
        contour_xy=np.zeros((0, 2), dtype=np.float32),
        normals_xy=np.zeros((0, 2), dtype=np.float32),
    )

    m = (plan_mask_u8 > 127).astype(np.uint8) * 255
    if not m.any():
        return empty

    k = _odd_kernel_size(kernel_px)
    if k >= 3:
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, el, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, el, iterations=1)

    fg_before_cc = int((m > 0).sum())
    if fg_before_cc == 0:
        return empty

    largest, largest_area = _largest_component_mask(m)
    if largest_area == 0:
        return empty

    if min_area_frac > 0.0 and largest_area < min_area_frac * float(fg_before_cc):
        return empty

    if boundary_blur_sigma > 1e-6:
        sig = float(boundary_blur_sigma)
        ksz = max(3, 2 * int(np.ceil(3.0 * sig)) + 1)
        ksz = ksz | 1
        seg = largest.astype(np.float32) / 255.0
        blurred = cv2.GaussianBlur(seg, (ksz, ksz), sig)
        thr = (blurred >= 0.5).astype(np.uint8) * 255
        mask_for_contour, mc_area = _largest_component_mask(thr)
        if mc_area == 0:
            mask_for_contour = largest
    else:
        mask_for_contour = largest

    contours, _ = cv2.findContours(mask_for_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return OutlineResult(
            mask_clean_u8=mask_for_contour,
            contour_xy=np.zeros((0, 2), dtype=np.float32),
            normals_xy=np.zeros((0, 2), dtype=np.float32),
        )
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1.0:
        return OutlineResult(
            mask_clean_u8=mask_for_contour,
            contour_xy=np.zeros((0, 2), dtype=np.float32),
            normals_xy=np.zeros((0, 2), dtype=np.float32),
        )

    xy = cnt.reshape(-1, 2).astype(np.float32)
    if epsilon_poly and epsilon_poly > 0.0:
        eps = float(epsilon_poly)
        arc = cv2.arcLength(cnt, closed=True)
        simp = cv2.approxPolyDP(cnt, eps * arc, closed=True)
        xy = simp.reshape(-1, 2).astype(np.float32)

    if contour_smooth_sigma_pts > 1e-6:
        if contour_preserve_straight_edges:
            xy = smooth_contour_straight_segments_preserve_corners(
                xy,
                float(contour_smooth_sigma_pts),
                corner_angle_deg=float(contour_corner_angle_deg),
                edge_straighten_max_dev_px=float(edge_straighten_max_dev_px),
            )
        else:
            xy = smooth_closed_polyline_xy(xy, float(contour_smooth_sigma_pts))

    normals = _contour_normals_unit(xy)
    return OutlineResult(mask_clean_u8=mask_for_contour, contour_xy=xy, normals_xy=normals)


def draw_outline_overlay_bgr(
    panel_bgr: np.ndarray,
    outline: OutlineResult,
    *,
    contour_color: Tuple[int, int, int] = (0, 255, 255),
    contour_thickness: int = 1,
    draw_normals: bool = True,
    normal_step: int = 8,
    normal_len_px: float = 10.0,
    normal_color: Tuple[int, int, int] = (255, 128, 0),
) -> np.ndarray:
    """
    Copy ``panel_bgr`` and draw contour (and optionally subsampled normals) for visualization.
    """
    out = panel_bgr.copy()
    c = outline.contour_xy
    if c.shape[0] >= 2:
        pts = np.round(c).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, contour_color, contour_thickness, lineType=cv2.LINE_AA)
    if draw_normals and c.shape[0] >= 2 and outline.normals_xy.shape[0] == c.shape[0]:
        step = max(1, int(normal_step))
        nrm = outline.normals_xy
        for i in range(0, len(c), step):
            x0, y0 = float(c[i, 0]), float(c[i, 1])
            nx, ny = float(nrm[i, 0]), float(nrm[i, 1])
            x1 = int(round(x0 + normal_len_px * nx))
            y1 = int(round(y0 + normal_len_px * ny))
            cv2.line(
                out,
                (int(round(x0)), int(round(y0))),
                (x1, y1),
                normal_color,
                1,
                lineType=cv2.LINE_AA,
            )
    return out
