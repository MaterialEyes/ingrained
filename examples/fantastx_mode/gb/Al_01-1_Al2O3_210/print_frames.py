import sys
sys.path.append('../../../../')
import numpy as np
import matplotlib.pyplot as plt
import ingrained.image_ops as iop
from ingrained.utilities import print_frames

# For preprocessing experimental image
from skimage.transform import rescale
from skimage import restoration
from skimage.exposure import equalize_adapthist

# Read image data
image_data = iop.image_open('HAADF39.dm3')

# Prepare experimental image (make sure this procedure matches the procedure in 'run.py')
exp_img = iop.apply_rotation(image_data['Pixels'],1)[271-10:783+10,0:520]
exp_img = iop.scale_pixels(exp_img, mode='rescale')
exp_img = restoration.wiener(exp_img, np.ones((7, 7))/3.5,1300)
exp_img = equalize_adapthist(exp_img ,clip_limit=0.005)

# Print the first 10 frames to file
# print_frames(poscar_file='bicrystal.POSCAR.vasp', exp_img=exp_img, exp_title="", progress_file="progress.txt", frame_selection="1-30", search_mode="gb", cmap="gray")

# Print the best frame to file
print_frames(poscar_file='bicrystal.POSCAR.vasp', exp_img=exp_img, exp_title="", progress_file="progress.txt", frame_selection="best", search_mode="gb", cmap="gray")