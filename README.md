# Non-parametric Calibration for Classification

<p align="center">
  <img src="figures/gpcalib_illustration/latent_process.png" alt="latent_process" width="512"/>
</p>

This repository provides the implementation of our paper ["Non-parametric Calibration for Classification"](addlink) (Jonathan Wenger, Hedvig Kjellström, Rudolph Triebel). All results presented in our work were produced with this code.

* [Introduction](#introduction)
* [Installation](#usage)
* [Datasets](#data)
* [Experiments](#experiments)
* [Publication](#paper)
* [License and Contact](#other)


## <a name="usage">Introduction</a>


## <a name="usage">Installation</a>
The code was developed in python 3.6 under macOS Mojave (10.14). You can clone the repo with:
```
git clone https://github.com/JonathanWenger/pycalib
```
or install it directly the python package directly via:
```
pip install git+https://github.com/JonathanWenger/pycalib.git
```

## <a name="data">Datasets</a>

* PCam

Due to the size of the data, only a script replicating the experiments is provided. The data can be downloaded from the [PCam repository](https://github.com/basveeling/pcam).

* KITTI

The repository includes 64-dimensional features extracted from KITTI sequences compressed in a zip file (data/kitti_features.zip).

* MNIST

A script will automatically download the MNIST dataset if an experiment is run on it.

* ImageNet 2012

Due to the size of the data, only a script replicating the experiments is provided. The ImageNet validation data can be obtained http://www.image-net.org.


## <a name="experiments">Experiments</a>

The repository includes scripts that replicate the experiments found in the paper, including:

* TODO

## <a name="paper">Publication</a>
If you use this code in your work, please cite the following paper.

Jonathan Wenger, Hedvig Kjellström and Rudolph Triebel, _"Non-parametric Calibration for Classification"_, in ... ([pdf](addlink))


    @InProceedings{wenger2019calibration,
      author = "J. Wenger and H. Kjellström and R. Triebel",
      title = "Non-parametric Calibration for Classification",
      booktitle = "",
      year = "2019",
      month = "October",
      keywords={calibration, non-parametric, gaussian processes, object classification},
      note = {{<a href="https://github.com/JonathanWenger/pycalib" target="_blank">[code]</a>} },
    }

## <a name="other"> License and Contact</a>

This work is released under the [MIT License].

Contact **Jonathan Wenger** [:envelope:](mailto:j.wenger@tum.de) for questions and comments. Please submit an issue for reporting bugs.
