import sys
sys.path.append('../../../../')
import numpy as np
import matplotlib.pyplot as plt
import ingrained.image_ops as iop
from ingrained.utilities import prepare_fantastx_input

# For preprocessing experimental image
from skimage.transform import rescale
from skimage import restoration
from skimage.exposure import equalize_adapthist

# Read image data
image_data = iop.image_open('HAADF149.dm3')

# Prepare experimental image (make sure this procedure matches the procedure in 'run.py')
exp_img = image_data['Pixels'][0:470,0:470]

prepare_fantastx_input(poscar_file='bicrystal.POSCAR.vasp', exp_img=exp_img, progress_file="progress.txt")