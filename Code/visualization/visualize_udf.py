import argparse
import os
import sys
import numpy as np
import torch
from pyhocon import ConfigFactory


def setup_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def load_model(config_path, checkpoint_path, device="cuda:0"):
    setup_project_root()
    from models.fields import SREUDFNetwork

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        device = "cpu"
    device = torch.device(device)

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        conf = ConfigFactory.parse_string(f.read())

    model = SREUDFNetwork(**conf["model"]["udf_network"]).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("udf_network_fine", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the configured SREUDFNetwork.\n"
            f"Missing keys: {missing[:10]}\nUnexpected keys: {unexpected[:10]}\n"
            "Use the exact configuration used during training."
        )
    model.eval()
    print(f"Loaded model from: {checkpoint_path}")
    print(f"Device: {device}")
    return model, conf, device


def load_point_cloud(path):
    """Load .npy, .npz, .xyz/.txt/.pts, or common mesh/point-cloud formats."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Point-cloud file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        data = np.load(path)
        key = "points" if "points" in data else list(data.keys())[0]
        arr = data[key]
    elif ext in (".xyz", ".txt", ".pts", ".csv"):
        arr = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
    else:
        try:
            import trimesh
            obj = trimesh.load(path, process=False)
            if isinstance(obj, trimesh.Scene):
                obj = trimesh.util.concatenate(
                    [g for g in obj.geometry.values() if hasattr(g, "vertices")]
                )
            if hasattr(obj, "vertices"):
                arr = np.asarray(obj.vertices)
            elif hasattr(obj, "vertices") is False and hasattr(obj, "points"):
                arr = np.asarray(obj.points)
            else:
                raise ValueError("Loaded object does not contain vertices/points.")
        except Exception as exc:
            raise ValueError(
                f"Could not read {path}. Use .npy, .npz, .xyz, .txt, .pts, .csv, .ply, or .obj."
            ) from exc

    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected point data with shape [N, >=3], got {arr.shape}")
    arr = arr[:, :3].astype(np.float32)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) == 0:
        raise ValueError(f"No valid 3D points found in {path}")
    return arr


def get_bounds(points, padding=0.05, bbox_min=None, bbox_max=None):
    if bbox_min is not None and bbox_max is not None:
        lo = np.asarray(bbox_min, dtype=np.float32)
        hi = np.asarray(bbox_max, dtype=np.float32)
    else:
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        span = np.maximum(hi - lo, 1e-4)
        lo = lo - span * padding
        hi = hi + span * padding
    if lo.shape != (3,) or hi.shape != (3,) or np.any(hi <= lo):
        raise ValueError("Invalid bounding box. Provide three bbox_min and bbox_max values.")
    return lo, hi


def query_udf_and_gradient(model, points, device, batch_size=16384, need_gradient=True):
    """Returns UDF [N] and optionally coordinate gradients [N,3]."""
    values, gradients = [], []
    for start in range(0, len(points), batch_size):
        p = torch.as_tensor(points[start:start + batch_size], dtype=torch.float32, device=device)
        p.requires_grad_(need_gradient)
        with torch.enable_grad():
            d = model.udf(p)
            values.append(d.detach().reshape(-1).cpu().numpy())
            if need_gradient:
                g = torch.autograd.grad(
                    outputs=d,
                    inputs=p,
                    grad_outputs=torch.ones_like(d),
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )[0]
                gradients.append(g.detach().cpu().numpy())
    vals = np.concatenate(values)
    grads = np.concatenate(gradients) if need_gradient else None
    return vals, grads


def add_common_args(parser):
    parser.add_argument("--config", required=True, help="Training-time .conf file")
    parser.add_argument("--checkpoint", required=True, help="Trained .pth checkpoint")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Visualize a 2D UDF slice with gradient arrows."
    )
    add_common_args(parser)
    parser.add_argument("--pointcloud", required=True,
                        help="Reference point cloud used to estimate the plotting bounds")
    parser.add_argument("--output", default="visualization/results/udf_gradient.png")
    parser.add_argument("--resolution", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--arrow-step", type=int, default=12)
    parser.add_argument("--slice-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--slice-value", type=float, default=None,
                        help="Fixed coordinate; default is the midpoint of the point-cloud bounds")
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--near-surface-percentile", type=float, default=3.0,
                        help="Draw contour at this percentile of sampled UDF values (approximation)")
    args = parser.parse_args()

    if args.resolution < 16 or args.arrow_step < 1:
        raise ValueError("--resolution must be >=16 and --arrow-step >=1")

    model, conf, device = load_model(args.config, args.checkpoint, args.device)
    points = load_point_cloud(args.pointcloud)
    lo, hi = get_bounds(points, padding=args.padding)

    axis_idx = {"x": 0, "y": 1, "z": 2}[args.slice_axis]
    plane_axes = [i for i in range(3) if i != axis_idx]
    a, b = plane_axes
    fixed = float((lo[axis_idx] + hi[axis_idx]) / 2) if args.slice_value is None else args.slice_value
    if not lo[axis_idx] <= fixed <= hi[axis_idx]:
        print("Warning: slice value lies outside the point-cloud bounds.")

    ca = np.linspace(lo[a], hi[a], args.resolution, dtype=np.float32)
    cb = np.linspace(lo[b], hi[b], args.resolution, dtype=np.float32)
    A, B = np.meshgrid(ca, cb, indexing="xy")
    xyz = np.zeros((A.size, 3), dtype=np.float32)
    xyz[:, a], xyz[:, b], xyz[:, axis_idx] = A.ravel(), B.ravel(), fixed

    Dflat, Gflat = query_udf_and_gradient(
        model, xyz, device, batch_size=args.batch_size, need_gradient=True
    )
    D = Dflat.reshape(args.resolution, args.resolution)
    G = Gflat.reshape(args.resolution, args.resolution, 3)
    Ga, Gb = G[:, :, a], G[:, :, b]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.contourf(A, B, D, levels=60, cmap="turbo")
    finite = D[np.isfinite(D)]
    if finite.size:
        eps = float(np.percentile(finite, args.near_surface_percentile))
        if np.nanmin(D) <= eps <= np.nanmax(D):
            ax.contour(A, B, D, levels=[eps], colors="black", linewidths=1.2)

    mag = np.sqrt(Ga * Ga + Gb * Gb)
    Ua = Ga / np.maximum(mag, 1e-8)
    Ub = Gb / np.maximum(mag, 1e-8)
    sl = (slice(None, None, args.arrow_step), slice(None, None, args.arrow_step))
    ax.quiver(
        A[sl], B[sl], Ua[sl], Ub[sl], color="white",
        angles="xy", scale_units="xy", scale=max(args.arrow_step * 2.5, 1),
        width=0.002, alpha=0.9
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Unsigned distance")
    labels = ["X", "Y", "Z"]
    ax.set_xlabel(labels[a])
    ax.set_ylabel(labels[b])
    ax.set_title(f"UDF and gradient field ({args.slice_axis}={fixed:.4f})")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {args.output}")
    print(f"UDF range: [{float(np.nanmin(D)):.6g}, {float(np.nanmax(D)):.6g}]")
    print("Note: the black contour is a low-distance percentile contour, not necessarily the exact zero surface.")


if __name__ == "__main__":
    main()
