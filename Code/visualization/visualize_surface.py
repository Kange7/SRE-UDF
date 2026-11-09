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
    import trimesh

    parser = argparse.ArgumentParser(
        description="Sample a trained UDF on a 3D grid, extract a low-distance isosurface, and render it."
    )
    add_common_args(parser)
    parser.add_argument("--pointcloud", required=True, help="Reference point cloud to define bounds")
    parser.add_argument("--output-dir", default="visualization/results/surface")
    parser.add_argument("--resolution", type=int, default=160,
                        help="Grid resolution per axis; memory/time grows cubically")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--threshold", type=float, default=None,
                        help="UDF is nonnegative: default uses a low percentile of sampled grid values")
    parser.add_argument("--threshold-percentile", type=float, default=1.0,
                        help="Percentile used to choose default approximate isosurface threshold")
    parser.add_argument("--max-grid-points", type=int, default=8_000_000,
                        help="Safety cap on number of grid samples")
    args = parser.parse_args()

    if args.resolution < 16:
        raise ValueError("--resolution must be >=16")
    total = args.resolution ** 3
    if total > args.max_grid_points:
        raise ValueError(
            f"Grid has {total:,} points, exceeding --max-grid-points={args.max_grid_points:,}. "
            "Use a lower --resolution or explicitly raise the safety cap."
        )

    model, conf, device = load_model(args.config, args.checkpoint, args.device)
    points = load_point_cloud(args.pointcloud)
    lo, hi = get_bounds(points, padding=args.padding)
    axes = [
        np.linspace(lo[i], hi[i], args.resolution, dtype=np.float32)
        for i in range(3)
    ]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)

    values = []
    for start in range(0, len(grid), args.batch_size):
        p = torch.as_tensor(grid[start:start + args.batch_size],
                            dtype=torch.float32, device=device)
        with torch.no_grad():
            d = model.udf(p)
        values.append(d.detach().reshape(-1).cpu().numpy())
        if (start // args.batch_size) % 50 == 0:
            print(f"Queried {min(start + args.batch_size, len(grid)):,}/{len(grid):,} grid points")
    field = np.concatenate(values).reshape(
        args.resolution, args.resolution, args.resolution
    )

    if args.threshold is None:
        threshold = float(np.percentile(field, args.threshold_percentile))
    else:
        threshold = float(args.threshold)
    if not np.isfinite(threshold) or threshold <= float(np.nanmin(field)):
        raise ValueError(
            f"Threshold {threshold} must be greater than the minimum sampled UDF "
            f"({float(np.nanmin(field))}). Try --threshold-percentile 1 or set --threshold manually."
        )
    if threshold >= float(np.nanmax(field)):
        raise ValueError("Threshold is at or above the maximum field value; choose a smaller threshold.")

    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise ImportError("Install scikit-image: pip install scikit-image") from exc

    spacing = tuple((hi - lo) / (args.resolution - 1))
    verts, faces, normals, _ = marching_cubes(
        field.astype(np.float32), level=threshold, spacing=spacing
    )
    verts += lo[None, :]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    os.makedirs(args.output_dir, exist_ok=True)
    mesh_path = os.path.join(args.output_dir, "udf_approx_surface.ply")
    mesh.export(mesh_path)

    # Matplotlib render: useful for quick inspection and paper drafts.
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    # Plot a decimated mesh surface for a lightweight preview.
    tri = mesh.triangles
    max_faces = 45000
    if len(tri) > max_faces:
        ids = np.linspace(0, len(tri) - 1, max_faces).astype(int)
        tri = tri[ids]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    collection = Poly3DCollection(
        tri, facecolor="lightgray", edgecolor="none", alpha=1.0
    )
    ax.add_collection3d(collection)
    ax.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Approximate UDF isosurface (threshold={threshold:.5g})")
    ax.set_box_aspect(np.maximum(hi - lo, 1e-6))
    fig.tight_layout()
    fig_path = os.path.join(args.output_dir, "udf_approx_surface.png")
    fig.savefig(fig_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    np.save(os.path.join(args.output_dir, "udf_grid.npy"), field)
    print(f"Saved mesh: {mesh_path}")
    print(f"Saved render: {fig_path}")
    print(f"UDF range: [{float(np.nanmin(field)):.6g}, {float(np.nanmax(field)):.6g}]")
    print(f"Extraction threshold: {threshold:.6g}")
    print(
        "Note: this is an approximate level set of a nonnegative UDF, not a sign-changing SDF zero crossing. "
        "Tune --threshold or --threshold-percentile and inspect the mesh for artifacts."
    )


if __name__ == "__main__":
    main()
