
# Hong Kong Landslide Susceptibility Mapping in a Meta-learning Way (tf2).
# Landslide Susceptibility Assessment in Multiple Landslide-inducing Environments with a Landslide Inventory Augmented by InSAR Techniques

##Table of Contents

- [Background](#background)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Contact](#contact)


## Background
LSA

<img src="figs/Overview.jpg" width="800px" hight="800px"/> 

​         Fig. 1: Overflow


## Dependencies

This code is implemented with the anaconda environment:
* cudatoolkit 11.2.2
* cudnn 8.1.0.77
* gdal 3.2.3
* numpy 1.23.3
* pandas 1.5.0
* python 3.9.13
* scikit-learn 1.1.2
* tensorflow 2.10.0
* tqdm 4.64.1

[//]: # (## Data)

[//]: # ()
[//]: # (The source and experiment data will be opened...)


## Usage

* For the unsupervised pretraining stage, see `./Unsupervised Pretraining/DAS_pretraining.py` and pretrain the base model. The parameter would be saved in `./unsupervised_pretraining/model_init/savedmodel.npz`.
* For the scene segmentation and task sampling stage, see `./scene_sampling.py`, the result would be output into `./metatask_sampling` folder.
* For the meta learner, see `./meta_learner.py`.
* For the model adaption and landslide susceptibility prediction, see `./predict_LSM.py`. The intermediate model and adapted models of blocks would be saved in folder `./checkpoint_dir` and `./models_of_blocks`, respectively.The adapted models will predict the susceptibility for each sample vector in `./src_data/grid_samples_HK.csv`.
* The `./tmp` folder restores some temp records.
* For the figuring in the experiment, see `./figure.py`, the figures would be saved in folder `./figs`.


## Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/CLi-de/Meta_LSM).

