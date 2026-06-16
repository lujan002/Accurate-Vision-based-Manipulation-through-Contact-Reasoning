"""Tests for classical top-down outline (no SAM2 / PyTorch import)."""

from __future__ import annotations

import numpy as np
import pytest

from topdown_outline import (
    fuse_outline_masks_image_plane,
    keep_fused_components_touching_reference,
    outline_from_topdown_mask,
    prune_outline_history_masks,
    smooth_closed_polyline_xy,
    smooth_contour_straight_segments_preserve_corners,
)


def test_empty_mask_returns_empty_outline() -> None:
    m = np.zeros((64, 64), dtype=np.uint8)
    r = outline_from_topdown_mask(m)
    assert r.contour_xy.shape == (0, 2)
    assert r.normals_xy.shape == (0, 2)
    assert not r.mask_clean_u8.any()


def test_rectangle_with_speckle_noise() -> None:
    """Filled rectangle with isolated salt noise; largest CC + morphology should preserve a simple contour."""
    h, w = 120, 100
    m = np.zeros((h, w), dtype=np.uint8)
    m[30:90, 25:75] = 255
    rng = np.random.default_rng(0)
    for _ in range(40):
        py, px = int(rng.integers(0, h)), int(rng.integers(0, w))
        if m[py, px] == 0:
            m[py, px] = 255

    r = outline_from_topdown_mask(m, kernel_px=3)
    assert r.contour_xy.shape[0] >= 4
    assert r.normals_xy.shape == r.contour_xy.shape
    # Normals are unit length (when defined)
    norms = np.linalg.norm(r.normals_xy, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    # Contour stays near the rectangle boundary (center of mass inside the box)
    cx, cy = r.contour_xy[:, 0].mean(), r.contour_xy[:, 1].mean()
    assert 40 < cx < 60 and 50 < cy < 70
    # Clean mask recovers most of the rectangle area
    assert r.mask_clean_u8.sum() / 255.0 > 0.45 * (60 * 50)


def test_min_area_frac_requires_large_enough_largest_component() -> None:
    """If largest CC is too small relative to total foreground, reject (anti-noise)."""
    h, w = 50, 50
    m = np.zeros((h, w), dtype=np.uint8)
    m[5:8, 5:8] = 255  # 3x3 = 9
    m[40:48, 40:48] = 255  # 8x8 = 64
    r1 = outline_from_topdown_mask(m, kernel_px=1, min_area_frac=0.95)
    assert r1.contour_xy.shape[0] == 0
    assert not r1.mask_clean_u8.any()


def test_approx_poly_reduces_points() -> None:
    m = np.zeros((80, 80), dtype=np.uint8)
    m[20:60, 20:60] = 255
    r_dense = outline_from_topdown_mask(m, kernel_px=1, epsilon_poly=0.0)
    r_simp = outline_from_topdown_mask(m, kernel_px=1, epsilon_poly=0.02)
    assert r_simp.contour_xy.shape[0] <= r_dense.contour_xy.shape[0]
    assert r_simp.contour_xy.shape[0] >= 4


def test_invalid_mask_shape() -> None:
    with pytest.raises(ValueError, match="2D"):
        outline_from_topdown_mask(np.zeros((5, 5, 1), dtype=np.uint8))


def test_fuse_outline_fills_gap_from_past_and_anchor() -> None:
    h, w = 40, 40
    past = np.zeros((h, w), dtype=np.uint8)
    past[10:30, 10:30] = 255
    current = np.zeros((h, w), dtype=np.uint8)
    current[10:30, 10:18] = 255  # occluded on the right vs past
    fused = fuse_outline_masks_image_plane(
        current,
        [past],
        None,
        out_h=h,
        out_w=w,
        fuse_anchor=True,
        clip_past_and_anchor_to_dilated_current=True,
        fusion_dilate_radius_px=0,
    )
    assert fused[15, 25] > 200  # filled from past
    anchor = np.zeros((h, w), dtype=np.uint8)
    anchor[5:8, 5:8] = 255
    fused2 = fuse_outline_masks_image_plane(
        np.zeros((h, w), dtype=np.uint8),
        [],
        anchor,
        out_h=h,
        out_w=w,
        fuse_anchor=True,
        clip_past_and_anchor_to_dilated_current=True,
    )
    assert fused2[6, 6] > 200


def test_clip_excludes_distant_past_but_legacy_or_keeps_both() -> None:
    h, w = 80, 80
    past = np.zeros((h, w), dtype=np.uint8)
    past[10:25, 10:25] = 255
    current = np.zeros((h, w), dtype=np.uint8)
    current[50:65, 50:65] = 255
    fused_clipped = fuse_outline_masks_image_plane(
        current,
        [past],
        None,
        out_h=h,
        out_w=w,
        fuse_anchor=True,
        clip_past_and_anchor_to_dilated_current=True,
        fusion_dilate_radius_px=5,
    )
    assert fused_clipped[17, 17] < 200
    assert fused_clipped[57, 57] > 200
    fused_legacy = fuse_outline_masks_image_plane(
        current,
        [past],
        None,
        out_h=h,
        out_w=w,
        fuse_anchor=True,
        clip_past_and_anchor_to_dilated_current=False,
        cc_filter_touching_current=False,
    )
    assert fused_legacy[17, 17] > 200
    assert fused_legacy[57, 57] > 200


def test_boundary_blur_and_contour_smooth_keep_valid_outline() -> None:
    m = np.zeros((100, 100), dtype=np.uint8)
    m[40:60, 40:60] = 255
    r1 = outline_from_topdown_mask(
        m, kernel_px=1, boundary_blur_sigma=2.0, contour_smooth_sigma_pts=3.0
    )
    assert r1.contour_xy.shape[0] >= 4
    norms = np.linalg.norm(r1.normals_xy, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_smooth_closed_polyline_preserves_square_topology() -> None:
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)
    s = smooth_closed_polyline_xy(xy, sigma_pts=1.5)
    assert s.shape == xy.shape
    assert np.all(np.isfinite(s))


def test_corner_preserving_smoothing_stays_closer_to_rect_corners() -> None:
    """Noisy axis-aligned square: preserve-corners path should land nearer true corners than circular smooth."""
    n_edge = 40
    pts: list[list[float]] = []
    for x in np.linspace(0, 100, n_edge, endpoint=False):
        pts.append([float(x), 0.0 + 0.8 * np.sin(x * 0.3)])
    for y in np.linspace(0, 100, n_edge, endpoint=False):
        pts.append([100.0 + 0.8 * np.sin(y * 0.25), float(y)])
    for x in np.linspace(100, 0, n_edge, endpoint=False):
        pts.append([float(x), 100.0 + 0.8 * np.cos(x * 0.2)])
    for y in np.linspace(100, 0, n_edge, endpoint=False):
        pts.append([0.0 + 0.8 * np.cos(y * 0.22), float(y)])
    xy = np.array(pts, dtype=np.float32)
    circ = smooth_closed_polyline_xy(xy, sigma_pts=8.0)
    pres = smooth_contour_straight_segments_preserve_corners(
        xy, 8.0, corner_angle_deg=18.0, edge_straighten_max_dev_px=3.5
    )
    true_corner = np.array([100.0, 0.0], dtype=np.float32)
    kc = int(np.argmin(np.linalg.norm(circ - true_corner[None, :], axis=1)))
    kp = int(np.argmin(np.linalg.norm(pres - true_corner[None, :], axis=1)))
    d_circ = float(np.linalg.norm(circ[kc] - true_corner))
    d_pres = float(np.linalg.norm(pres[kp] - true_corner))
    assert d_pres < d_circ * 0.92


def test_keep_fused_components_drops_blob_not_touching_current() -> None:
    cur = np.zeros((60, 60), dtype=bool)
    cur[40:55, 40:55] = True
    fused = np.zeros((60, 60), dtype=bool)
    fused[40:55, 40:55] = True
    fused[5:15, 5:15] = True  # ghost blob
    out = keep_fused_components_touching_reference(fused, cur)
    assert out[10, 10] == False
    assert out[47, 47] == True


def test_empty_current_anchor_clipped_to_dilated_last_past() -> None:
    h, w = 50, 50
    past_last = np.zeros((h, w), dtype=np.uint8)
    past_last[20:40, 20:40] = 255
    anchor = np.zeros((h, w), dtype=np.uint8)
    anchor[5:8, 5:8] = 255  # far from past_last
    fused = fuse_outline_masks_image_plane(
        np.zeros((h, w), dtype=np.uint8),
        [past_last],
        anchor,
        out_h=h,
        out_w=w,
        fuse_anchor=True,
        clip_past_and_anchor_to_dilated_current=True,
        fusion_dilate_radius_px=3,
        cc_filter_touching_current=False,
    )
    assert fused[30, 30] > 200
    assert fused[6, 6] < 200


def test_prune_outline_history_drops_moved_pose() -> None:
    h, w = 100, 100
    cur = np.zeros((h, w), dtype=np.uint8)
    cur[60:85, 60:85] = 255
    old = np.zeros((h, w), dtype=np.uint8)
    old[10:30, 10:30] = 255
    good = np.zeros((h, w), dtype=np.uint8)
    good[62:84, 62:84] = 255
    pruned = prune_outline_history_masks(
        [old, good],
        cur,
        h,
        w,
        iou_min=0.02,
        max_centroid_px=15.0,
    )
    assert len(pruned) == 1
    assert pruned[0].sum() == good.sum()
