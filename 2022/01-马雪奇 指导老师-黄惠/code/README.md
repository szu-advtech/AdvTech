# NDC
PyTorch implementation for paper [Neural Dual Contouring](https://arxiv.org/abs/2202.01999), [Zhiqin Chen](https://czq142857.github.io/), [Andrea Tagliasacchi](https://taiya.github.io/), [Thomas Funkhouser](https://www.cs.princeton.edu/~funk/), [Hao Zhang](http://www.cs.sfu.ca/~haoz/).

### [Paper](https://arxiv.org/abs/2202.01999)

## Citation
If you find Zhiqin's work useful in your research, please consider citing:

	@article{chen2022ndc,
	  title={Neural Dual Contouring}, 
	  author={Zhiqin Chen and Andrea Tagliasacchi and Thomas Funkhouser and Hao Zhang},
	  journal={ACM Transactions on Graphics (Special Issue of SIGGRAPH)},
	  volume = {41},
	  number = {4},
	  year={2022}
	}

## Requirements
- Python 3 with numpy, h5py, scipy, scikit-learn, trimesh, and Cython （I use Pyhton=3.7）
- [PyTorch 1.8](https://pytorch.org/get-started/locally/) (other versions may also work)

Build Cython module:
```
python setup.py build_ext --inplace
```

## Datasets and pre-trained weights
For data preparation, please see [data_preprocessing](https://github.com/czq142857/NDC/tree/master/data_preprocessing).

## Testing on one shape

You can use the code to test a trained model on one shape. The following shows example commands for testing with the pre-trained network weights and example shapes provided above.

Basically, ```--test_input``` specifies the input file. It could be a grid of SDF or UDF values, a grid of binary occupancies, or a point cloud. The supported formats can be found in the example commands below. ```--input_type``` specifies the input type corresponding to the input file. ```--method``` specifies the method to be applied. It could be NDC, UNDC, or NDCx. NDCx is basically NDC with a more complex backbone network from our prior work [Neural Marching Cubes (NMC)](https://github.com/czq142857/NMC) ; it is slower than NDC but has better reconstruction accuracy. ```--postprocessing``` indicates the result will be post-processed to remove small holes; it can only be applied to UNDC outputs.

To test on point cloud input:
```
python main.py --test_input examples/mobius.ply --input_type pointcloud --method undc --postprocessing --point_num 1024 --grid_size 64
python main.py --test_input examples/tshirt.ply --input_type pointcloud --method undc --postprocessing --point_num 8192 --grid_size 128
```
Note that ```--point_num``` specifies the maximum number of input points; if the input file contains more points than the specified number, the point cloud will be sub-sampled. ```--grid_size``` specifies the size of the output grid.

To test on a large scene (noisy point cloud input):
```
python main.py --test_input examples/E9uDoFAP3SH_region31.ply --input_type noisypc --method undc --postprocessing --point_num 524288 --grid_size 64 --block_num_per_dim 10
```
Note that the code will crop the entire scene into overlapping patches. ```--point_num``` specifies the maximum number of input points per patch. ```--grid_size``` specifies the size of the output grid per patch. ```--block_padding``` controls the boundary padding for each patch to make the patches overlap with each other so as to avoid seams; the default value is good enough in most cases. ```--block_num_per_dim``` specifies how many crops the scene will be split into. In the above command, the input point cloud will be normalized into a cube and the cube will be split into 10x10x10 patches (although some patches are empty).

## Training

To train/test UNDC with point cloud input:
```
python main.py --train_bool --input_type pointcloud --method undc --epoch 250 --lr_half_life 100 --data_dir ./groundtruth/gt_UNDC --checkpoint_save_frequency 10 --point_num 4096 --grid_size 64
python main.py --train_float --input_type pointcloud --method undc --epoch 250 --lr_half_life 100 --data_dir ./groundtruth/gt_UNDC --checkpoint_save_frequency 10 --point_num 4096 --grid_size 64
```

To train UNDC with noisy point cloud input, you need to prepare the augmented training data, see instructions in [data_preprocessing](https://github.com/czq142857/NDC/tree/master/data_preprocessing). Then run the following commands for training.
```
python main.py --train_bool --input_type noisypc --method undc --epoch 20 --lr_half_life 5 --data_dir ./groundtruth/gt_UNDCa --checkpoint_save_frequency 1 --point_num 16384 --grid_size 64
python main.py --train_float --input_type noisypc --method undc --epoch 20 --lr_half_life 5 --data_dir ./groundtruth/gt_UNDCa --checkpoint_save_frequency 1 --point_num 16384 --grid_size 64
```

