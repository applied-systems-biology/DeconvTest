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

1. Download [Fiji](https://fiji.sc/#download), make sure that the Fiji installation path does not contain spaces; note the Fiji installation path: you will have to provide it later when installing the DeconvTest packge
1. Download the [DeconvolutionLab_2.jar](http://bigwww.epfl.ch/deconvolution/deconvolutionlab2/) and [Iterative_Deconvolve_3D.class](https://imagej.net/Iterative_Deconvolve_3D) and copy them to the plugins folder of Fiji
1. Install [python 2.7 Anaconda](https://www.anaconda.com/distribution/)
1. Download the [DeconvTest](https://github.com/applied-systems-biology/DeconvTest/releases) package, enter the package directory and install the package by running ``python setup.py install`` withing the Anaconda environment
1. Provide the path to Fiji when prompted

## License

The source code of this framework is released under the <a href="/LICENSE">3-clause BSD license</a>

## Documentation

The DeconvTest package contains several subpackages.

- `classes`: implementation of all classes needed to conveniently work with synthetic image data.
- `modules`: functions to generate synthetic images, deconvolve and quantify the data.
- `batch`: functions to run the framework in a batch mode running multiple processes in parallel.
- `fiji`: ImageJ macros to run the integrated Fiji plugins.
- `scripts`: exemplary scripts and configuration files.
- `tests`: unit tests.

For a detailed user guide, 
see the <a href="https://applied-systems-biology.github.io/DeconvTest/">online documentation</a>
