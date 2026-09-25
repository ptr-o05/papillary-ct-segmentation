# Phantom tests for the Slicer-independent core of papillary_segmentation.py.
# Run with:  python -m pytest tests/

import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import papillary_segmentation as ps  

VOX = 0.75
N = 112


def _cylinder(grid, p0, p1, r_mm):
    zz, yy, xx = grid
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = np.linalg.norm(d)
    d /= length
    P = np.stack([zz - p0[0], yy - p0[1], xx - p0[2]], -1)
    t = P @ d
    perp = np.linalg.norm(P - t[..., None] * d, axis=-1)
    return (t >= 0) & (t <= length) & (perp * VOX <= r_mm)


def make_phantom(seed=0):
    # Ellipsoidal LV cavity with two papillary muscles, trabeculae and an RV
    rng = np.random.default_rng(seed)
    grid = np.mgrid[:N, :N, :N].astype(float)
    zz, yy, xx = grid
    c = N // 2
    lv = (((zz - c) * VOX / 32) ** 2 + ((yy - c) * VOX / 22) ** 2 + ((xx - c) * VOX / 22) ** 2) <= 1.0

    def wall_point(theta_deg, z=c, frac=1.0):
        th = np.radians(theta_deg)
        return np.array([z, c + frac * 22 / VOX * np.sin(th), c + frac * 22 / VOX * np.cos(th)])

    pm1 = _cylinder(grid, wall_point(200, frac=1.02), wall_point(200, frac=0.45), 4.5) & lv
    pm2 = _cylinder(grid, wall_point(320, frac=1.02), wall_point(320, frac=0.45), 4.0) & lv
    trab = np.zeros_like(lv)
    for _ in range(20):
        th, z0 = rng.uniform(0, 360), rng.uniform(c - 20, c + 20)
        a = wall_point(th, z=z0)
        b = wall_point(th + rng.uniform(-25, 25), z=z0 + rng.uniform(-12, 12), frac=0.8)
        trab |= _cylinder(grid, a, b, 1.0)
    trab &= lv
    img = np.full((N, N, N), 60.0)
    img[lv] = 380.0
    img[pm1 | pm2 | trab] = 90.0
    img += rng.normal(0, 8, img.shape)
    rv_center = np.array([c, c, c + 36 / VOX])
    rv = np.linalg.norm(np.stack([zz - rv_center[0], yy - rv_center[1], xx - rv_center[2]], -1), axis=-1) * VOX <= 14
    return img, lv, rv, pm1, pm2, trab


def _dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def test_split_recovers_both_muscles_and_labels_them():
    img, lv, rv, pm1, pm2, trab = make_phantom()
    lv_center = np.array(ndimage.center_of_mass(lv))
    septal = np.array(ndimage.center_of_mass(rv)) - lv_center

    tissue = ps.threshold_lv_tissue(img, lv)
    res = ps.split_papillary_muscles(tissue, lv, VOX, septal)

    assert res["split_method"] == "multiscale_opening"
    assert res["labelling_method"] == "septal_angle"
    # The muscle closer to the septal direction is the PPM
    ang = lambda m: ps._angle_deg(np.array(ndimage.center_of_mass(m)) - lv_center, septal)  # noqa: E731
    gt_ppm, gt_apm = (pm1, pm2) if ang(pm1) < ang(pm2) else (pm2, pm1)
    assert _dice(res["apm"], gt_apm) > 0.9
    assert _dice(res["ppm"], gt_ppm) > 0.9
    assert res["ppm_angle_to_septum_deg"] < res["apm_angle_to_septum_deg"]
    # Free trabeculae (not part of either muscle) contribute < 2 % of the muscle volume
    muscles = res["apm"] | res["ppm"]
    leak = (muscles & trab & ~(pm1 | pm2)).sum()
    assert leak < 0.02 * muscles.sum()


def test_fallback_path_and_missing_rv():
    img, lv, rv, pm1, pm2, trab = make_phantom()
    tissue = ps.threshold_lv_tissue(img, lv)
    saved = ps.find_papillary_cores
    try:
        ps.find_papillary_cores = lambda *a, **k: None
        res = ps.split_papillary_muscles(tissue, lv, VOX, None)
    finally:
        ps.find_papillary_cores = saved
    assert res["split_method"] == "distance_peaks_fallback"
    assert res["labelling_method"] == "image_axis_fallback"
    assert res["apm"].any() and res["ppm"].any()


def test_residual_blood_pool_is_detected():
    img, lv, *_ = make_phantom()
    tissue = ps.threshold_lv_tissue(img, lv, hu_max=1000)
    try:
        ps.split_papillary_muscles(tissue, lv, VOX, None)
    except ps.SplitError as e:
        assert "blood pool" in str(e)
    else:
        raise AssertionError("SplitError expected")


def test_threshold_bounds_are_inclusive():
    t = ps.threshold_lv_tissue(np.array([[-190, 242, 243, -191]]), np.ones((1, 4)), -190, 242)
    assert t.tolist() == [[True, True, False, False]]
