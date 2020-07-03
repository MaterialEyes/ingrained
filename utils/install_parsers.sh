#!/bin/bash
# for pySPM, might need to run: 
pip install pySPM

# for dm3_lib, might need to run: 
git clone https://bitbucket.org/piraynal/pydm3reader/;
cd pydm3reader/;
pip install .;
cd ..;
rm -rf pydm3reader;