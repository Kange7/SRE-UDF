
## Installation

- Install python Dependencies
```bash
conda create -n sreudf python=3.8
conda activate sreudf
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tqdm pyhocon==0.3.57 trimesh PyMCubes scipy point_cloud_utils==0.29.7
```
- Compile C++ extensions
```
cd extensions/chamfer_dist
python setup.py install
```


## Train
You can train our SRE-UDF to reconstruct surfaces from a single point cloud as:

```
python run.py --gpu 0 --conf confs/sewer3d.conf --dataname real_484 --dir real_484
```

You can find the generated mesh and the log in `./outs`.

## Test
You can evaluate the reconstructed meshes and dense point clouds as follows:


```
python evaluation/Sewer3d/eval_mesh.py --conf confs/sewer3d.conf --dataname real_484 --dir real_484
```


For evaluating the generated dense point clouds, you can run the `eval_pc.py` of each dataset instead of `eval_mesh.py`. 

## Use Your Own Data
We also provide the instructions for training your own data in the following.

### Data
First, you should put your own data to the `./data/owndata/input` folder. The datasets is organised as follows:
```
│data/
├──owndata/
│  ├── input
│      ├── (dataname).ply/xyz/npy
```
We support the point cloud data format of `.ply`, `.xyz` and `.npy`

### Run
To train your own data, simply run:
```
python run.py --gpu 0 --conf confs/base.conf --dataname (dataname) --dir (dataname)
```

