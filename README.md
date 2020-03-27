# AeroRIT: A New Scene for Hyperspectral Image Analysis
<img src="pagecontent/image_rgb.jpg" width="512">

This scene overlooking Rochester Institute of Technology was captured via Headwall Photonics Micro Hy-perspec E-Series CMOS sensor at an altitude of approximately 5,000 feat (0.4m GSD). The semantic map pixels were labelled with ENVI 4.8.2, using individual hyperspectral signatures and the geo-registered RGB images as references. 

## Structure of the repository
This repository is organized as:
* [Aerial Data](/Aerial%20Data/) This directory contains all the data required for training and analysis.
* [configs](/configs/) This directory contains a few examples for config files used in train and test functions.
* [helpers](/helpers/) This directory the helper zoo for non-model related functions.
* [networks](/networks/) This directory the model zoo used in the paper.
* [savedmodels](/savedmodels/) This directory will contain all the saved models post training.

## Dataset

The scene images can be found [here](https://drive.google.com/drive/folders/1yCMqa9uDC_CEGtbnxeWEQCTb-odC2r4c?usp=sharing). The directory contains four files: 
1. image_rgb - The RGB rectified hyperspectral scene.
2. image_hsi_radiance - Radiance calibrated hyperspectral scene sampled at every 10th band (400nm, 410nm, 420nm, .. 900nm).
3. image_hsi_reflectance - Reflectance calibrated hyperspectral scene sampled at every 10th band.
4. image_labels - Semantic labels for the entire AeroCampus scene.

Note: The above files only contain every 10th band from 400nm to 900nm. You can request for the full size versions of both radiance and reflectance via an [email](mailto:aneesh.rangnekar@mail.rit.edu?subject=[GitHub]%20AeroCampus%20Full%20Version) to the corresponding author.

## Baselines Performance on the dataset
We train a series of networks with weighted cross-entropy loss and report the performance in terms of mean Intersection-over-Union: 

| Model | mIOU |
| -- | -- |
| SegNet | 52.60 |
| SegNet-m | 59.08 |
| Res-U-Net (6) | 72.55 |
| Res-U-Net (9) | 70.88 |
| U-Net | 60.40 |
| U-Net-m | 70.62 |
| U-Net-m + SE | 75.35 |
| U-Net-m + SE + PReLU act. | 75.89 |
| U-Net-m + SE + PReLU act. + Self Supervised Learning | **76.40** |

All pretrained models can be found [here](https://drive.google.com/drive/folders/1n7hwc4D05OIpmuPSIsuYOrfT4PDFk6tS?usp=sharing). For re-checking or improving network architecture, please place the model files into [savedmodels](/savedmodels/) before executing any code. 

## Requirements

numpy 

cv2 (opencv-python)

pytorch1.0.1

Pillow

We recommend to use [Anaconda](https://www.anaconda.com/distribution/) environment for running all sets of code. We have tested our code on Ubuntu 16.04 with Python 3.6.

## Executing codes

Before running any files, execute [sampling_data.py](/sampling_data.py/) to obtain train, validation and test splits with 64 x 64 image chips. 

Some of the important arguments used in [train](/train.py/) and [test](/test.py/) files are as follows:

| Argument | Description |
| -- | -- |
| config-file | path to configuration file if present |
| bands | how many bands to sample from HSI imagery (3 -> RGB, 51 -> all) ? |
| hsi_c | use HSI radiance or reflectance for analysis ? |
| network_arch | which network architecture to use: Res-U-Net, SegNet or U-Net? |
| network_weights_path | path to save(d) network weights |
| use_cuda | use GPUs for processing or CPU? |

Please refer to the corresponding files for indepth argument descriptions. To cross-verify the final reported result in the paper, [test.py](/test.py/) should be run with as follows:
`python test.py --config configs/eval-unetm-best.yaml`

## License

This scene dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
1. That the dataset comes “AS IS”, without express or implied warranty. Although every effort has been made to ensure accuracy, we (Rochester Institute of Technology) do not accept any responsibility for errors or omissions.
2. That you include a reference to the AeroCampus Dataset in any work that makes use of the dataset.
3. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
4. You may not use the dataset or any derivative work for commercial purposes such as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
5. That all rights not expressly granted to you are reserved by us (Rochester Institute of Technology).

## Citation

When using the dataset or code, please cite our [paper](https://arxiv.org/pdf/1912.08178.pdf): 
```
@misc{rangnekar2019aerorit,
    title={AeroRIT: A New Scene for Hyperspectral Image Analysis},
    author={Aneesh Rangnekar and Nilay Mokashi and Emmett Ientilucci and Christopher Kanan and Matthew J. Hoffman},
    year={2019},
    eprint={1912.08178},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```

## Acknowledgements

The codebase is heavily based off [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg). Both are great repositories - have a look!


