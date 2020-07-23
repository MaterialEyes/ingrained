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
image_data = iop.image_open('HAADF26.dm3')

# Prepare experimental image (make sure this procedure matches the procedure in 'run.py')
exp_img = np.fliplr(iop.apply_crop(iop.apply_rotation(image_data['Pixels'],66),328,328)[:-26,:-26])

# Print the best frame to file
print_frames(poscar_file='bicrystal.POSCAR.vasp', exp_img=exp_img, exp_title="", progress_file="progress.txt", frame_selection="best", search_mode="gb", cmap="gray")