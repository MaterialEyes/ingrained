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
image_data = iop.image_open('HAADF85.dm3')

# Prepare experimental image (make sure this procedure matches the procedure in 'run.py')
exp_img = image_data['Pixels']
exp_img = rescale(exp_img, 3.0, anti_aliasing=True)
exp_img = restoration.wiener(exp_img, np.ones((5, 5)) / 50, 60)
exp_img = equalize_adapthist(exp_img ,clip_limit=0.008)
exp_img = iop.apply_rotation(exp_img,1.975)
exp_img = rescale(exp_img, 0.6, anti_aliasing=True)
exp_img = exp_img[280:-120,180:-120]

prepare_fantastx_input(poscar_file='bicrystal.POSCAR.vasp', exp_img=exp_img, progress_file="progress.txt")