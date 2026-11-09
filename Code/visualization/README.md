# CAPUDF visualization scripts

These scripts load an existing SREUDFNetwork checkpoint and do not train or update model weights.

## Folder layout
Place this `visualization/` folder directly under your project root, next to `run.py` and `models/`.
The training-time `.conf` and `.pth` checkpoint can remain in their existing locations.

Install plotting/extraction dependencies in the same Python environment as the project:
```bash
pip install matplotlib trimesh scikit-image
```
`pyhocon`, PyTorch, and the project's `models/fields.py` are also required.

## Examples
Run commands from the project root and replace paths with your actual files.

```bash
python visualization/visualize_udf.py --config ./confs/ndscene.conf --checkpoint ./PATH/ckpt_XXXXX.pth --pointcloud ./PATH/input.xyz --output visualization/results/udf_gradient.png
python visualization/visualize_movement.py --config ./confs/ndscene.conf --checkpoint ./PATH/ckpt_XXXXX.pth --pointcloud ./PATH/input.xyz --output-dir visualization/results/movement --steps 1
python visualization/visualize_surface.py --config ./confs/ndscene.conf --checkpoint ./PATH/ckpt_XXXXX.pth --pointcloud ./PATH/input.xyz --output-dir visualization/results/surface --resolution 160
```

## Input formats
Point cloud inputs: `.npy`, `.npz` (uses `points` key if available), `.xyz`, `.txt`, `.pts`, `.csv`, `.ply`, `.obj`.
The text formats should contain at least three numeric columns: x y z.

## Important notes
- Use the exact configuration used to train the checkpoint.
- All scripts infer the plotting bounds from the supplied point cloud, with configurable padding. Ensure that the point cloud and model use the same coordinate system and normalization.
- UDF values are nonnegative. `visualize_udf.py` displays a low-percentile distance contour, not a guaranteed exact surface.
- `visualize_surface.py` extracts an approximate level set at a positive UDF threshold. Threshold and grid resolution affect surface quality.
- `visualize_movement.py` follows the first-order projection rule used in the supplied `run.py`; it is a visualization/inference procedure, not an optimization/training step.
- For larger scenes, reduce `--resolution` first. A cubic grid can become expensive quickly.
