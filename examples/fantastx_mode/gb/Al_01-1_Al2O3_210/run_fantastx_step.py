import os
import sys
import numpy as np
sys.path.append('../../../../')
import ingrained.image_ops as iop
import matplotlib.pyplot as plt
from ingrained.structure import Bicrystal

# --------------------------------------------------------------------
# These are loaded at the beginning of a FANTASTX run and remain fixed
# --------------------------------------------------------------------

# Read experimental image from 'fantastx_start' (read this same image for every fantastx iteration)
exp_img = np.load('fantastx_start/experiment.npy')

# Read simulation parameters from 'fantastx_start', optimize using ingrained image fusion 
sim_params = iop.load_sim_params('fantastx_start/sim_params')

# If you want to view the cropped experimental image
plt.imshow(exp_img,cmap='gray'); plt.axis('off'); plt.title('experiment'); plt.show();

# --------------------------------------------------------------------
# These files are modified for each FANTASTX iteration (in a loop)
# --------------------------------------------------------------------

# Filename for the current poscar
current_poscar = 'fantastx_start/ingrained.POSCAR.vasp'

# Initialize a Bicrystal object from a poscar file, directly!
bicrystal = Bicrystal(poscar_file=current_poscar);

# Simulate an image 
sim_img, __ = bicrystal.simulate_image(sim_params=sim_params)

print(sim_params)

# View the image before proceeding with optimization
plt.imshow(sim_img,cmap='gray'); plt.axis('off'); plt.title('simulation'); plt.show();

objective = iop.score_ssim(sim_img, exp_img)

print("FOM: {}".format(objective))

# --------------------------------------------------------------------
# This is only run the 1st time to ensure initial structure is best
# --------------------------------------------------------------------

# Check that objective for initial structure equals best objective from 'progress.txt'
progress_file = 'progress.txt'
if os.path.isfile('progress.txt'):
    progress = np.genfromtxt(progress_file, delimiter=',')
    best_idx = int(np.argmin(progress[:,-1]))
    x = progress[best_idx]

assert objective == x[-1]
print("Initial structure is 'ingrained' optimal!")