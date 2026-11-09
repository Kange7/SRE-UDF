import argparse
import os
import numpy as np
import torch
import trimesh
from pyhocon import ConfigFactory
from scipy.spatial import cKDTree

from mesh_evaluator import MeshEvaluator

gt_cache = {}



def estimate_normals(points, k=30):

    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

  
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        tree = cKDTree(points)
        d, _ = tree.query(points, k=2)
        mean_spacing = float(np.mean(d[:, 1])) if len(d) > 0 else 1.0
        radius = max(mean_spacing * 3.0, 1e-6)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=k
            )
        )
        pcd.normalize_normals()
        return np.asarray(pcd.normals, dtype=np.float32)
    except ImportError:
        pass


    tree = cKDTree(points)
    k = min(k, len(points))
    _, idx = tree.query(points, k=k)              # (N, k)
    neigh = points[idx]                            # (N, k, 3)
    neigh_centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', neigh_centered, neigh_centered) / k
    eigvals, eigvecs = np.linalg.eigh(cov)         
    normals = eigvecs[:, :, 0]                     
    return normals.astype(np.float32)


def _sanitize_normals(normals):
    
    if normals is None:
        return None
    normals = np.asarray(normals, dtype=np.float32).copy()
    if normals.size == 0:
        return normals
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    bad = (~np.isfinite(normals).all(axis=-1)) | (lengths[..., 0] < 1e-12)
    if bad.any():
        normals[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    return normals


def load_gt(gt_path, n_points):
    
    ext = os.path.splitext(gt_path)[-1].lower()
    gt_normals = None

    if ext == '.xyz':
        gt_pointcloud = np.loadtxt(gt_path).astype(np.float32)
    else:
        gt_mesh = trimesh.load_mesh(gt_path)

        if isinstance(gt_mesh, trimesh.Trimesh):
            gt_pointcloud, idx = gt_mesh.sample(n_points, return_index=True)
            gt_pointcloud = gt_pointcloud.astype(np.float32)
            gt_normals = np.asarray(gt_mesh.face_normals[idx], dtype=np.float32)
            gt_normals = _sanitize_normals(gt_normals)

        elif isinstance(gt_mesh, trimesh.PointCloud):
            verts = np.asarray(gt_mesh.vertices, dtype=np.float32)
            
            vnorm = getattr(gt_mesh, 'vertex_normals', None)
            perm = np.random.permutation(len(verts))
            gt_pointcloud = verts[perm][:n_points]
            if vnorm is not None and len(vnorm) == len(verts):
                gt_normals = np.asarray(vnorm, dtype=np.float32)[perm][:n_points]
                gt_normals = _sanitize_normals(gt_normals)
            else:
                gt_normals = None

        else:
            raise RuntimeError(f'Unknown GT data type: {type(gt_mesh)} ({gt_path})')

    
    if gt_normals is None:
        print('[eval_mesh] GT has no normals — estimating them (Open3D / PCA).')
        gt_normals = _sanitize_normals(estimate_normals(gt_pointcloud))

    return gt_pointcloud, gt_normals


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def evaluate_mesh(gt_path, es_path, input_path, n_points=100000, normalize=True):
    es_mesh = trimesh.load_mesh(es_path)
    gt_scale = 1.

    inputshape = trimesh.load(input_path)
    total_size = (inputshape.bounds[1] - inputshape.bounds[0]).max()
    centers = (inputshape.bounds[1] + inputshape.bounds[0]) / 2
    es_mesh.apply_scale(total_size)
    es_mesh.apply_translation(centers)

    # ---- GT ----
    if gt_path in gt_cache:
        gt_pointcloud, gt_normals, gt_scale = gt_cache[gt_path]
    else:
        gt_pointcloud, gt_normals = load_gt(gt_path, n_points)

        if normalize:
            center = gt_pointcloud.mean(0)
            gt_scale = np.abs(gt_pointcloud - center).max()
            print('Normalize with scale: %.2f' % gt_scale)

        gt_pointcloud = gt_pointcloud / gt_scale
        gt_cache[gt_path] = (gt_pointcloud, gt_normals, gt_scale)

    scale = np.abs(gt_pointcloud).max()
    if normalize:
        es_mesh.vertices /= gt_scale
        scale = 1.

    thresholds = np.linspace(1. / 1000, 1, 1000) * scale
    result = evaluator.eval_mesh(
        es_mesh, gt_pointcloud, gt_normals, thresholds=thresholds
    )

    print('Chamfer-L1:', result['chamfer-L1'] * 10)
    print('Chamfer-L2:', result['chamfer-L2'] * 10)
    print('F-score@5%:', result['f-score-5'])
    print('F-score@10%:', result['f-score-10'])
    print('Normal Consistency:', result['normals'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/ndf.conf')
    parser.add_argument('--mcube_resolution', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dir', type=str, default='test')
    parser.add_argument('--dataname', type=str, default='demo')
    parser.add_argument('--pred_mesh', type=str, default='demo')
    args = parser.parse_args()

    with open(args.conf) as f:
        conf_text = f.read()
    conf = ConfigFactory.parse_string(conf_text)

    evaluator = MeshEvaluator(n_points=50000)

    gt_file = os.path.join(
        conf.get_string('dataset.data_dir'), 'ground_truth', args.dataname + '.ply'
    )
    pred_file = os.path.join(
        conf.get_string('general.base_exp_dir'), args.dir, 'mesh',
        str(conf.get_int('train.step2_maxiter')) + '_mesh.obj'
    )
    input_file = os.path.join(
        conf.get_string('dataset.data_dir'), 'input', args.dataname + '.ply'
    )

    evaluate_mesh(gt_file, pred_file, input_file, 150000)
