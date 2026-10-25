"""Estimate an RGB-to-NIR planar homography from corresponding pixel points.

--points JSON format: {"rgb": [[x,y], ...], "nir": [[x,y], ...]}.
Without --points, explicitly opens two photo windows for paired clicks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np

LIMITATION = "Planar homography: valid only for the calibrated plane/depth and unchanged camera setup; no hardware synchronization is established."


def image_size(value):
    a = np.asarray(value)
    if a.shape != (2,) or not np.all(np.isfinite(a)) or np.any(a <= 0) or np.any(a != np.round(a)):
        raise ValueError("Image sizes must be positive integer [width,height]")
    return [int(a[0]), int(a[1])]


def validate_matrix(matrix):
    h = np.asarray(matrix, dtype=np.float64)
    if h.shape != (3, 3) or not np.all(np.isfinite(h)) or abs(h[2, 2]) < 1e-12:
        raise ValueError("Homography must be a finite 3x3 matrix with nonzero normalization")
    h = h / h[2, 2]
    condition = float(np.linalg.cond(h))
    if abs(float(np.linalg.det(h))) < 1e-12 or not np.isfinite(condition) or condition > 1e10:
        raise ValueError("Homography is singular or ill-conditioned")
    return h, condition


def map_points(points, matrix):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or not np.all(np.isfinite(p)):
        raise ValueError("Points must be a finite Nx2 array in x,y pixel order")
    h, _ = validate_matrix(matrix)
    q = np.column_stack((p, np.ones(len(p)))) @ h.T
    if np.any(np.abs(q[:, 2]) < 1e-9):
        raise ValueError("Point projects onto the homography horizon")
    if np.any(q[:, 2] > 0) and np.any(q[:, 2] < 0):
        raise ValueError("Point group spans the homography horizon")
    result = q[:, :2] / q[:, 2, None]
    if not np.all(np.isfinite(result)) or np.max(np.abs(result), initial=0) > 1e9:
        raise ValueError("Projected coordinates are unstable")
    return result


def checked_points(points, size, label):
    points = np.asarray(points, dtype=np.float64)
    width, height = image_size(size)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4 or not np.all(np.isfinite(points)):
        raise ValueError(f"{label}: at least four finite x,y pixel points are required")
    if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
        raise ValueError(f"{label}: correspondence points are outside image bounds")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError(f"{label}: duplicate correspondence points")
    hull = cv2.convexHull(points.astype(np.float32))
    if cv2.contourArea(hull) < 1e-4 or np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-6) < 2:
        raise ValueError(f"{label}: points must not be collinear")
    return points


def estimate_calibration(rgb_points, nir_points, rgb_size_wh, nir_size_wh, threshold=3.0):
    if not np.isfinite(threshold) or not 0 < threshold <= 20:
        raise ValueError("RANSAC threshold must be positive and <=20 pixels")
    rgb = checked_points(rgb_points, rgb_size_wh, "RGB")
    nir = checked_points(nir_points, nir_size_wh, "NIR")
    if len(rgb) != len(nir):
        raise ValueError("RGB and NIR correspondence counts differ")
    h, mask = cv2.findHomography(rgb, nir, cv2.RANSAC, threshold)
    if h is None or mask is None:
        raise ValueError("Homography estimation failed")
    h, condition = validate_matrix(h)
    errors = np.linalg.norm(map_points(rgb, h) - nir, axis=1)
    inliers = mask.reshape(-1).astype(bool)
    if int(inliers.sum()) < 4 or float(inliers.mean()) < .5:
        raise ValueError("Too few consistent correspondences for calibration")
    checked_points(rgb[inliers], rgb_size_wh, "RGB inliers")
    checked_points(nir[inliers], nir_size_wh, "NIR inliers")
    if not np.all(np.isfinite(errors)) or errors[inliers].mean() > threshold or errors[inliers].max() > threshold * 2:
        raise ValueError("Calibration reprojection error is too large")
    return {"schema_version": 1, "direction": "rgb_to_nir", "matrix": h.tolist(),
            "rgb_size_wh": image_size(rgb_size_wh), "nir_size_wh": image_size(nir_size_wh),
            "rgb_points_xy": rgb.tolist(), "nir_points_xy": nir.tolist(),
            "inliers": inliers.tolist(), "fit_method": "cv2.findHomography RANSAC",
            "ransac_threshold_px": threshold,
            "reprojection": {"inlier_mean_error_px": float(errors[inliers].mean()),
                             "inlier_max_error_px": float(errors[inliers].max()),
                             "all_mean_error_px": float(errors.mean()),
                             "inlier_count": int(inliers.sum()), "point_count": len(rgb),
                             "condition_number": condition},
            "limitation": LIMITATION, "created_utc": datetime.now(timezone.utc).isoformat()}


def load_calibration(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if data.get("schema_version") != 1 or data.get("direction") != "rgb_to_nir":
        raise ValueError("Expected schema1 RGB-to-NIR calibration")
    h, condition = validate_matrix(data.get("matrix"))
    rgb = checked_points(data.get("rgb_points_xy"), data.get("rgb_size_wh"), "RGB")
    nir = checked_points(data.get("nir_points_xy"), data.get("nir_size_wh"), "NIR")
    if len(rgb) != len(nir):
        raise ValueError("Calibration point counts differ")
    mask = np.asarray(data.get("inliers"))
    if mask.shape != (len(rgb),) or not all(isinstance(x, bool) for x in data["inliers"]):
        raise ValueError("Invalid calibration inlier mask")
    if int(mask.sum()) < 4 or mask.mean() < .5:
        raise ValueError("Calibration has insufficient inliers")
    checked_points(rgb[mask], data["rgb_size_wh"], "RGB inliers")
    checked_points(nir[mask], data["nir_size_wh"], "NIR inliers")
    threshold = float(data.get("ransac_threshold_px", 3))
    if not np.isfinite(threshold) or not 0 < threshold <= 20:
        raise ValueError("Invalid calibration threshold")
    errors = np.linalg.norm(map_points(rgb, h) - nir, axis=1)
    if errors[mask].mean() > threshold or errors[mask].max() > threshold * 2:
        raise ValueError("Calibration matrix disagrees with saved correspondences")
    data["reprojection"] = {"inlier_mean_error_px": float(errors[mask].mean()),
                            "inlier_max_error_px": float(errors[mask].max()),
                            "all_mean_error_px": float(errors.mean()),
                            "inlier_count": int(mask.sum()), "point_count": len(rgb),
                            "condition_number": condition}
    data["matrix"] = h.tolist()
    data["limitation"] = LIMITATION
    return data


def verify_dimensions(calibration, rgb_size_wh, nir_size_wh):
    if image_size(rgb_size_wh) != image_size(calibration["rgb_size_wh"]) or image_size(nir_size_wh) != image_size(calibration["nir_size_wh"]):
        raise ValueError("Current RGB/NIR frame dimensions differ from calibration; recalibrate, do not automatically scale")


def click_correspondences(rgb, nir):
    """Explicit interactive photo-only flow; this function is never used by tests."""
    points = {"RGB": [], "NIR": []}
    images = {"RGB": rgb, "NIR": nir}
    titles = {name: f"Calibration {name} - same point order; Enter finish; Esc abort" for name in points}
    def clicked(event, x, y, flags, name):
        if event == cv2.EVENT_LBUTTONDOWN:
            points[name].append([x, y])
    try:
        for name in points:
            cv2.namedWindow(titles[name], cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(titles[name], clicked, name)
        while True:
            for name in points:
                preview = images[name].copy()
                for i, (x, y) in enumerate(points[name]):
                    cv2.circle(preview, (x, y), 4, (0, 255, 255), -1)
                    cv2.putText(preview, str(i + 1), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                cv2.imshow(titles[name], preview)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or any(cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1 for title in titles.values()):
                raise KeyboardInterrupt("Calibration aborted; no file saved")
            if key in (10, 13) and len(points["RGB"]) == len(points["NIR"]) and len(points["RGB"]) >= 4:
                return points["RGB"], points["NIR"]
    finally:
        cv2.destroyAllWindows()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rgb", type=Path, required=True)
    p.add_argument("--nir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("calibration.json"))
    p.add_argument("--points", type=Path, help="Paired pixel coordinates; supplied => no GUI opens")
    p.add_argument("--ransac-threshold", type=float, default=3.0)
    args = p.parse_args(argv)
    from run_inspection import read_photo
    rgb, nir = read_photo(args.rgb), read_photo(args.nir)
    if args.output.resolve() in {args.rgb.resolve(), args.nir.resolve(), args.points.resolve() if args.points else None}:
        p.error("Calibration output must differ from input files")
    if args.points:
        points = json.loads(args.points.read_text(encoding="utf-8-sig"))
        rp, np_ = points["rgb"], points["nir"]
    else:
        print("Click corresponding points in the same order in both photos, at least four; Enter finishes, Esc aborts.")
        rp, np_ = click_correspondences(rgb, nir)
    data = estimate_calibration(rp, np_, [rgb.shape[1], rgb.shape[0]], [nir.shape[1], nir.shape[0]], args.ransac_threshold)
    data.update(rgb_photo=str(args.rgb.resolve()), nir_photo=str(args.nir.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved: {args.output.resolve()}")
    print(f"Inlier reprojection mean: {data['reprojection']['inlier_mean_error_px']:.3f}px")
    print(LIMITATION)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Calibration aborted; no file saved", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
