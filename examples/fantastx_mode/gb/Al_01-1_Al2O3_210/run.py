import sys
sys.path.append('../../../../')
import numpy as np
import ingrained.image_ops as iop
import matplotlib.pyplot as plt
from ingrained.structure import Bicrystal
from ingrained.optimize import CongruityBuilder

# For preprocessing experimental image
from skimage.transform import rescale
from skimage import restoration
from skimage.exposure import equalize_adapthist

# Read image data
image_data = iop.image_open('HAADF39.dm3')

# Constrain optimization to clean region of image by cropping (and cleaning)
exp_img = iop.apply_rotation(image_data['Pixels'],1)[271-10:783+10,0:520]
exp_img = iop.scale_pixels(exp_img, mode='rescale')
exp_img = restoration.wiener(exp_img, np.ones((7, 7))/3.5,1300)
exp_img = equalize_adapthist(exp_img ,clip_limit=0.005)

# View the image before proceeding with optimization
plt.imshow(exp_img,cmap='gray'); plt.axis('off'); plt.show();

# Initialize a Bicrystal object and save the constructed bicrystal structure
# bicrystal = Bicrystal(config_file='config.json', write_poscar=True);

# Structure was already created in a prior run, so loading directly is faster!
bicrystal = Bicrystal(poscar_file='bicrystal.POSCAR.vasp');

# Initialize a ConguityBuilder with the Bicrystal and experimental image
congruity = CongruityBuilder(sim_obj=bicrystal, exp_img=exp_img);

# Define initial set of input parameters for an image simulation
pix_size          = 0.1275
# pix_size          = image_data["Experiment Pixel Size"]
interface_width   = 0.00
defocus           = 1.00
x_shear           = 0.00
y_shear           = 0.00
x_stretch         = 0.00
y_stretch         = 0.00
crop_height       = 499
crop_width        = 185

sim_params = [pix_size, interface_width, defocus, x_shear, y_shear, x_stretch, y_stretch, crop_height, crop_width]

# Find correspondence!
congruity.find_correspondence(objective='taxicab_ssim', initial_solution=sim_params, search_mode="gb")