import logging
import numpy as np
import trimesh
from scipy.spatial import cKDTree


EMPTY_PCL_DICT = {
    'completeness': np.sqrt(3),
    'accuracy': np.sqrt(3),
    'completeness2': 3,
    'accuracy2': 3,
    'chamfer-L1': np.sqrt(3),
    'chamfer-L2': 3,
}

EMPTY_PCL_DICT_NORMALS = {
    'normals completeness': 0.,
    'normals accuracy': 0.,
    'normals': 0.,
}

logger = logging.getLogger(__name__)


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


def _nanmean_or_zero(x):

    if x is None:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    finite = np.isfinite(x)
    if not finite.any():
        return 0.0
    return float(x[finite].mean())


class MeshEvaluator(object):
    ''' Mesh evaluation class. '''

    def __init__(self, n_points=100000):
        self.n_points = n_points

    def eval_mesh(self, mesh, pointcloud_tgt, normals_tgt,
                  thresholds=np.linspace(1. / 1000, 1, 1000)):
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            pointcloud, idx = mesh.sample(self.n_points, return_index=True)
            pointcloud = pointcloud.astype(np.float32)
            normals = np.asarray(mesh.face_normals[idx], dtype=np.float32)
            normals = _sanitize_normals(normals)
        else:
            pointcloud = np.empty((0, 3), dtype=np.float32)
            normals = np.empty((0, 3), dtype=np.float32)

        out_dict = self.eval_pointcloud(
            pointcloud, pointcloud_tgt, normals, normals_tgt,
            thresholds=thresholds)
        return out_dict

    def eval_pointcloud(self, pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None,
                        thresholds=np.linspace(1. / 1000, 1, 1000)):
        if pointcloud.shape[0] == 0:
            logger.warning('Empty pointcloud / mesh detected!')
            out_dict = EMPTY_PCL_DICT.copy()
            out_dict.update(EMPTY_PCL_DICT_NORMALS)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # ---- Completeness: target -> predicted ----
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness ** 2

        completeness = float(completeness.mean())
        completeness2 = float(completeness2.mean())
        completeness_normals = _nanmean_or_zero(completeness_normals)

        # ---- Accuracy: predicted -> target ----
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy ** 2

        accuracy = float(accuracy.mean())
        accuracy2 = float(accuracy2.mean())
        accuracy_normals = _nanmean_or_zero(accuracy_normals)

        # ---- Chamfer distances ----
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        chamferL1 = 0.5 * (completeness + accuracy)

        # ---- Normal consistency ----
        normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals

        # ---- F-score ----
        F = [
            2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-12)
            for i in range(len(precision))
        ]

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1,

            'f-score-5':  F[4],   # threshold = 0.5%
            'f-score-10': F[9],   # threshold = 1%
        }
        return out_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt. '''
    kdtree = cKDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    has_normals = (
        normals_src is not None and normals_tgt is not None
        and len(normals_src) == len(points_src)
        and len(normals_tgt) == len(points_tgt)
        and len(normals_src) > 0 and len(normals_tgt) > 0
    )

    if has_normals:
        ns = _sanitize_normals(normals_src)
        nt = _sanitize_normals(normals_tgt)
        dots = (nt[idx] * ns).sum(axis=-1)
        dots = np.abs(dots)
        dots = np.where(np.isfinite(dots), dots, 0.0)
    else:
        
        dots = np.zeros(points_src.shape[0], dtype=np.float32)

    return dist, dots


def get_threshold_percentage(dist, thresholds):
    
    return [(dist <= t).mean() for t in thresholds]
