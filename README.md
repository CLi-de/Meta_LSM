# meta_LSM （Migrated from tf1 to tf2）
# Hong Kong Landslide Susceptibility Mapping in a Meta-learning Way.

##Table of Contents

- [Background](#background)
- [Dependencies](#dependencies)

[//]: # (- [Data]&#40;#data&#41;)
- [Usage](#usage)
- [Contact](#contact)


## Background
Landslide susceptibility assessment (LSA) is vital for landslide hazard mitigation and prevention. 
In recent years, the increased availability of high-quality satellite data and landslide statistics has promoted a wide range of applications of data-driven LSA methods. 
However, two issues are still concerned: (a) most landslide records from a landslide inventory (LI) are based on the interpretation of optical images and site investigation, leading data-driven model not sensitive to slope dynamics such as slow-moving landslides; 
(b) The study area usually contains a variety of landslide-inducing environments (LIEs) and can hardly be well expressed by a single model. 
Pointedly, we proposed to utilize InSAR techniques to sample from deformation slope for landslide inventory augmentation; and meta-learn intermediate model for fast adaption of multiple LIEs. 
We conducted feature permutation for various LIEs to identify dominant landslide-inducing factors (LIFs) and provided guidance for targeted landslide prevention schemes. 
The results in Hong Kong show that in some mountainous regions, the slow-moving slopes make up the majority of the recorded landslides. 
The discussion shows that slope and SPI are the most influential LIFs in Hong Kong. 
Compared to other data-driven LSA approaches, the highest statistical measure and fast adaption performance demonstrate the superiority and effectiveness of the proposed methods.

<img src="figs/overflow.pdf" width="800px" hight="800px"/> 

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

