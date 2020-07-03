import sys
sys.path.append('../../../../')
import numpy as np
import matplotlib.pyplot as plt
import ingrained.image_ops as iop
from ingrained.structure import Bicrystal
from ingrained.optimize import CongruityBuilder

# For preprocessing experimental image
from skimage.transform import rescale
from skimage import restoration
from skimage.exposure import equalize_adapthist

# Read image data
image_data = iop.image_open('HAADF85.dm3')

# Constrain optimization to clean region of image by cropping (and cleaning)
exp_img = image_data['Pixels']
exp_img = rescale(exp_img, 3.0, anti_aliasing=True)
exp_img = restoration.wiener(exp_img, np.ones((5, 5)) / 50, 60)
exp_img = equalize_adapthist(exp_img ,clip_limit=0.008)
exp_img = iop.apply_rotation(exp_img,2)
exp_img = rescale(exp_img, 0.5, anti_aliasing=True)
exp_img = exp_img[80::,80::]

# View the image before proceeding with optimization
plt.imshow(exp_img,cmap='gray'); plt.axis('off'); plt.show();

# Initialize a Bicrystal object with the path to the slab json file
bicrystal = Bicrystal(filename='config.json', write_poscar=True);

# Initialize a ConguityBuilder with Bicrystal and experimental image
congruity = CongruityBuilder(sim_obj=bicrystal, exp_img=exp_img);

# Input parameters to optimize for an image simulation:
pix_size          = 0.13125
interface_width   = 0.225
defocus           = 1.40
x_shear           = 0.00
y_shear           = 0.00
x_stretch         = 0.00
y_stretch         = 0.00
crop_height       = 213
crop_width        = 169

sim_params = [pix_size, interface_width, defocus, x_shear, y_shear, x_stretch, y_stretch, crop_height, crop_width]

# Find correspondence!
congruity.find_correspondence(objective='taxicab_ssim', initial_solution=sim_params, search_mode="gb")