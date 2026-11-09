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
        description="Project a point cloud toward the learned UDF surface and visualize movement."
    )
    add_common_args(parser)
    parser.add_argument("--pointcloud", required=True, help="Input point cloud")
    parser.add_argument("--output-dir", default="visualization/results/movement")
    parser.add_argument("--num-points", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=1,
                        help="Number of projection iterations")
    parser.add_argument("--max-step", type=float, default=None,
                        help="Optional maximum movement per iteration in coordinate units")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--arrow-count", type=int, default=800)
    args = parser.parse_args()

    if args.num_points < 1 or args.steps < 1:
        raise ValueError("--num-points and --steps must be positive")

    model, conf, device = load_model(args.config, args.checkpoint, args.device)
    source = load_point_cloud(args.pointcloud)
    rng = np.random.default_rng(args.seed)
    if len(source) > args.num_points:
        source = source[rng.choice(len(source), args.num_points, replace=False)]
    initial = source.copy()
    current = initial.copy()

    for step in range(args.steps):
        dist, grad = query_udf_and_gradient(
            model, current, device, batch_size=args.batch_size, need_gradient=True
        )
        norm = np.linalg.norm(grad, axis=1, keepdims=True)
        direction = grad / np.maximum(norm, 1e-8)
        delta = direction * dist[:, None]
        if args.max_step is not None:
            lengths = np.linalg.norm(delta, axis=1, keepdims=True)
            delta *= np.minimum(1.0, args.max_step / np.maximum(lengths, 1e-8))
        # Same first-order projection principle as in run.py.
        current = current - delta
        print(
            f"Step {step + 1}/{args.steps}: mean UDF before projection = "
            f"{float(np.mean(dist)):.6g}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    np.savetxt(os.path.join(args.output_dir, "points_before.xyz"), initial, fmt="%.7f")
    np.savetxt(os.path.join(args.output_dir, "points_after.xyz"), current, fmt="%.7f")

    # Color-coded 3D overview and movement vectors for a subset.
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(initial[:, 0], initial[:, 1], initial[:, 2],
                s=1.2, c="tab:blue", alpha=0.65)
    ax1.set_title("Before projection")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(current[:, 0], current[:, 1], current[:, 2],
                s=1.2, c="tab:orange", alpha=0.75)
    ax2.set_title("After projection")

    for ax in (ax1, ax2):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect(np.maximum(np.ptp(np.vstack([initial, current]), axis=0), 1e-6))

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "movement_before_after.png"),
                dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Separate plot with arrows, subsampled for legibility.
    _, grad0 = query_udf_and_gradient(
        model, initial, device, batch_size=args.batch_size, need_gradient=True
    )
    dist0, _ = query_udf_and_gradient(
        model, initial, device, batch_size=args.batch_size, need_gradient=False
    )
    moved = current - initial
    count = min(args.arrow_count, len(initial))
    ids = rng.choice(len(initial), count, replace=False) if count < len(initial) else np.arange(len(initial))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(initial[:, 0], initial[:, 1], initial[:, 2],
               s=1.0, c="lightgray", alpha=0.35, label="Input points")
    ax.scatter(current[:, 0], current[:, 1], current[:, 2],
               s=1.4, c="tab:green", alpha=0.8, label="Projected points")
    ax.quiver(
        initial[ids, 0], initial[ids, 1], initial[ids, 2],
        moved[ids, 0], moved[ids, 1], moved[ids, 2],
        color="tab:red", linewidth=0.5, arrow_length_ratio=0.2,
        normalize=False
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("UDF-guided point movement")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "movement_vectors.png"),
                dpi=250, bbox_inches="tight")
    plt.close(fig)

    np.savetxt(os.path.join(args.output_dir, "udf_before.txt"), dist0, fmt="%.8g")
    print(f"Saved movement outputs to: {args.output_dir}")
    print("Important: projection is a first-order approximation; inspect the output for unstable gradients or excessive movement.")


if __name__ == "__main__":
    main()
