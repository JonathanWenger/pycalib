# Non-parametric Calibration for Classification

<p align="center">
  <img src="figures/gpcalib_illustration/latent_process.png" alt="latent_process" width="512"/>
</p>

This repository provides the implementation of our paper ["Non-parametric Calibration for Classification"](addlink) (Jonathan Wenger, Hedvig Kjellström, Rudolph Triebel). All results presented in our work were produced with this code.

* [Introduction](#introduction)
* [Installation](#usage)
* [Documentation](#documentation)
* [Datasets and Experiments](#data)
* [Publication](#paper)
* [License and Contact](#other)


## <a name="usage">Introduction</a>

Many popular classification models in computer vision and robotics are often not calibrated, meaning their predicted uncertainties do not match the  probability of classifying correctly. This repository provides a new multi-class and model-agnostic approach to calibration, based on Gaussian processes, which have a number of desirable properties making them suitable as a calibration tool, such as the ability to incorporate prior knowledge.

## <a name="usage">Installation</a>
The code was developed in python 3.6 under macOS Mojave (10.14). You can clone the repo with
```
git clone https://github.com/JonathanWenger/pycalib
```
or install it directly the python package directly via
```
pip install git+https://github.com/JonathanWenger/pycalib.git
```

## <a name="usage">Documentation</a>

You can access the documentation of `pycalib` at https://jonathanwenger.github.io/pycalib/.

## <a name="data">Datasets and Experiments</a>

* PCam

Due to the size of the data, only a script replicating the experiments is provided. The data can be downloaded from the [PCam repository](https://github.com/basveeling/pcam).

* KITTI

The repository includes 64-dimensional features extracted from KITTI sequences compressed in a zip file (data/kitti_features.zip).

* MNIST

A script will automatically download the MNIST dataset if an experiment is run on it.

* ImageNet 2012

Due to the size of the data, only a script replicating the experiments is provided. The ImageNet validation data can be obtained from http://www.image-net.org.

The repository includes scripts that replicate the experiments found in the paper in the `benchmark` and `figures` folders.


## <a name="paper">Publication</a>
If you use this code in your work, please cite the following paper.

Jonathan Wenger, Hedvig Kjellström and Rudolph Triebel, _"Non-parametric Calibration for Classification"_, in ... ([pdf](addlink))


    @InProceedings{wenger2019calibration,
      author = "J. Wenger and H. Kjellström and R. Triebel",
      title = "Non-parametric Calibration for Classification",
      booktitle = "",
      year = "2019",
      keywords={calibration, non-parametric, gaussian processes, object classification},
      note = {{<a href="https://github.com/JonathanWenger/pycalib" target="_blank">[code]</a>}},
    }

## <a name="other"> License and Contact</a>

This work is released under the [MIT License].

Contact **Jonathan Wenger** [:envelope:](mailto:j.wenger@tum.de) for questions and comments. Please submit an issue for reporting bugs.
