# CPG
Object-Aware Spatial Constraint for Weakly Supervised Detection

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Installation](#installation)
4. [Usage](#usage)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```
  
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`
3. [Optional] MATLAB is required for **official** PASCAL VOC evaluation only. The code now includes unofficial Python evaluation code.

### Requirements: hardware

1. For training smaller networks (VGG_CNN_F, VGG_CNN_M_1024), a GPU with about 6G of memory suffices.
2. For training lager networks (VGG16), you'll need a GPU with about 8G of memory.

### Installation

1. Clone the CPG repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/shenyunhang/CPG.git
  ```

2. We'll call the directory that you cloned CPG into `CPG_ROOT`

   *Ignore notes 1 and 2 if you followed step 1 above.*

   **Note 1:** If you didn't clone CPG with the `--recursive` flag, then you'll need to manually clone the `caffe-wsl` submodule:
    ```Shell
    git submodule update --init --recursive
    ```
    **Note 2:** The `caffe-wsl` submodule needs to be on the `wsl` branch (or equivalent detached state). This will happen automatically *if you followed step 1 instructions*.

3. Build the Cython modules
    ```Shell
    cd $CPG_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $CPG_ROOT/caffe-wsl
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

5. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

6. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

7. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

8. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $CPG_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
9. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012
10. [Optional] If you want to use COCO, please see some notes under `data/README.md`
11. Follow the next sections to download pre-trained ImageNet models

### Download object proposals
1. Selective Search: [original matlab code](http://disi.unitn.it/~uijlings/MyHomepage/index.php#page=projects1), [python wrapper](https://github.com/sergeyk/selective_search_ijcv_with_python)
2. EdgeBoxes: [matlab code](https://github.com/pdollar/edges)
3. MCG: [matlab code](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/)


### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $CPG_ROOT
./data/scripts/fetch_imagenet_models.sh
```

### Usage

To train and test a CPG detector, use `experiments/scripts/cpg.sh`.
Output is written underneath `$CPG_ROOT/output`.

```Shell
cd $CPG_ROOT
./experiments/scripts/cpg.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {VGG_CNN_F, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify configure options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

Example:

```Shell
./experiments/scripts/cpg.sh 0 VGG16 pascal_voc --set EXP_DIR cpg
```

This will reproduction the VGG16 result in paper.

Trained CPG networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```

### Other method

WSDDN:

```Shell
./experiments/scripts/wsddn.sh 0 VGG16 pascal_voc --set EXP_DIR wsddn
```
or
```Shell
./experiments/scripts/wsddn_x.sh 0 VGG16 pascal_voc --set EXP_DIR wsddn_x
```

ContextLocNet:

```Shell
./experiments/scripts/contextlocnet.sh 0 VGG16 pascal_voc --set EXP_DIR contextlocnet
```
or
```Shell
./experiments/scripts/contextlocnet_x.sh 0 VGG16 pascal_voc --set EXP_DIR contextlocnet_x
```

