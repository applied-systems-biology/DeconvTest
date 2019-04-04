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

### Prerequisites
- python2.7
- setuptools
- pandas
- [HelperLib](https://github.com/applied-systems-biology/HelperLib)

To install the package run ``python setup.py install``.

To test the package run ``python setup.py test``.

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
