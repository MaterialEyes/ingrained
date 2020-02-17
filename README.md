![image](https://drive.google.com/uc?export=view&id=1H3PYFTqZpiytEAR4KFbFV_7U2bKUWyUV)

Grain boundary structure initialization from atomic-resolution HAADF STEM imaging
[[wiki](https://gitlab.com/MaterialEyes/ingrained/wikis/home)] [[paper](https://)]


## Getting started

### Requirements
You need a working python 3.x installation to be able to use ingrained, and [gdown](https://github.com/wkentaro/gdown) to download testing files and executables for STEM image simulation. We highly recommend using [Anaconda](https://anaconda.org/) to manage the installation environment.

### Installation
Clone this repo and create a new conda environment with Python 3.7 and gdown:
```sh
git clone https://github.com/eschwenk/ingrained
conda create -n ingrained -c conda-forge python=3.7 gdown 
```
Activate this environment, navigate to the root directory and download the files:
```sh
conda activate ingrained
cd ingrained
./bin/download.sh
```
Navigate to the [<code>structure.py</code>](https://github.com/MaterialEyes/ingrained/blob/master/ingrained/structure.py) module and set your [Materials Project API key](https://materialsproject.org/open) "MAPI_KEY" environment variable.
```python
os.environ['MAPI_KEY'] = "YOUR MP API Key"
```
Install package with pip:
```sh
pip install .
```
## Usage

Refer to the Jupyter notebook examples:
* [**CdTe–CdTe**](https://github.com/MaterialEyes/ingrained/blob/master/test/cdte-cdte_demo.ipynb)&nbsp;&nbsp;bicrystal (used to create initial structure optimized in [FIG. 4](https://aip.scitation.org/doi/10.1063/1.5123169))
* [**Al–Al<sub>2</sub>O<sub>3</sub>**](https://github.com/MaterialEyes/ingrained/blob/master/test/al-al2o3_demo.ipynb)&nbsp;&nbsp;bicrystal (need a [citation](https://))
* [**STM**](https://github.com/MaterialEyes/ingrained/blob/master/test/stm_demo.ipynb)&nbsp;&nbsp; (need a [citation](https://))
