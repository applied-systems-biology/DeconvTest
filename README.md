# DeconvTest: *in silico* microscopy framework to quantify deconvolution accuracy
Author: *Anna Medyukhina*

Affiliation: *Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge  
https://www.leibniz-hki.de/en/applied-systems-biology.html  
HKI-Center for Systems Biology of Infection  
Leibniz Institute for Natural Product Research and Infection Biology -  
Hans Knöll Insitute (HKI)  
Adolf-Reichwein-Straße 23, 07745 Jena, Germany*

---

DeconvTest is a python-based simulation framework that allows the user to quantify and compare the 
performance of different deconvolution methods. The framework integrates all components needed for such 
quantitative comparison and consists of three main modules: (1) *in silico* microscopy, 
(2) deconvolution, and (3) performance quantification. 

￼<img src="/docs/img/deconvtest_scheme.png" width="600">

## Installation

1. Install [Fiji](https://fiji.sc/#download); note the Fiji installation path: you will have to provide it later when installing the DeconvTest packge 
1. Download the [DeconvolutionLab_2.jar](http://bigwww.epfl.ch/deconvolution/deconvolutionlab2/) and [Iterative_Deconvolve_3D.class](https://imagej.net/Iterative_Deconvolve_3D) and copy them to the plugins folder of Fiji
1. Install [python 2.7 Anaconda](https://www.anaconda.com/distribution/)
1. Download [HelperLib package](https://github.com/applied-systems-biology/HelperLib)
1. Enter the HelperLib directory and install the HelperLib package by running ``python setup.py install`` withing the Anaconda environment
1. Download the DeconvTest package
1. Enter the DeconvTest directory and install the DeconvTest by running ``python setup.py install`` withing the Anaconda environment
1. Provide the path to Fiji when prompted

## License

The source code of this framework is released under the <a href="/LICENSE">3-clause BSD license</a>

## Documentation

The DeconvTest package contains several subpackages.

- `classes`: implementation of all classes needed to conveniently work with synthetic image data.
- `deconvolve`: interfaces to integrated ImageJ plugins.
- `batch`: implementation of the main modules of the framework:
    - `simulation`: the *in silico* microscopy module; contains functions for simulating a microscopy process 
    in a batch mode.
    - `deconvolution`: the deconvolution module; contains functions for deconvolving images with ImageJ plugins in 
    a batch mode. 
    - `quantification`: the performance quantification module; contains functions for evaluating deconvolution 
    accuracy in a batch mode.
- `scripts`: exemplary scripts and configuration files.
- `tests`: unit tests.

For a detailed user guide, 
see the <a href="https://applied-systems-biology.github.io/DeconvTest/">online documentation</a>
