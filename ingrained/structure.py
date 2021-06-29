import os,sys
import shutil
import random
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance, distance_matrix

import ingrained.image_ops as iop
from ingrained.construct import TopGrain, BottomGrain

from ase import Atom
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.io.vasp.inputs import Structure, Lattice
from pymatgen.io.vasp import Poscar

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

class PartialCharge(object):
    """
    This class contains the partial charge densities from a DFT-simulation
    with a calculator to simulate a STM image and perform postprocessing manipulations.
    """
    def __init__(self, config_file=""):
        """
        Initialize a PartialCharge object with the path to a PARCHG file.

        Args:
            config_file: (string) path to a PARCHG file
        """
        self.config_file = config_file
        self.parchg = Chgcar.from_file(config_file)

        # Actual volumetric charge data from PARCHG
        self.chgdata = self.parchg.data['total']

        # Structure associated w/ volumetric data
        self.structure = self.parchg.structure

        # Fraction of cell occupied by structure
        self.zmax = self.structure.cart_coords[:, 2].max()/self.structure.lattice.c

        # Get the real space sizes assocated with each voxel in volumetric data
        self.lwh = ( self.structure.lattice.a/self.parchg.dim[0] , \
                     self.structure.lattice.b/self.parchg.dim[1] , \
                     self.structure.lattice.c/self.parchg.dim[2] )

        # Image fields
        self.cell  = None
        self.image = None
        self.sim_params = None

    def _get_stm_vol(self, z_below="", z_above=""):
        """
        Get the volumetric slab associated with the input depths.

        Args:
            z_below: (float) thickness or depth from top in Angstroms
            z_above: (float) distance above the surface to consider

        Returns:
            A numpy array of the selected volumetric charge density slab.
            Indice of the top slice (in reference to full slab)
        """
        if self._parameter_check_z(z_below=z_below, z_above=z_above):

            # Get fractional coordinates for z_below and z_above
            z_below /= self.structure.lattice.c
            z_above /= self.structure.lattice.c

            # Get indices of volumetric slices to consider above and below the top surface (zmax)
            nzmin = int(self.parchg.dim[2] * (self.zmax - z_below))
            nzmax = int(self.parchg.dim[2] * (self.zmax + z_above))

            # Get volumetric slice (nzmin:nzmax,nrows,ncols) from charge data
            rhos  = np.swapaxes(np.swapaxes(self.chgdata.copy(),0,2),1,2)[nzmin:nzmax,:,:]/self.structure.volume
            return rhos, nzmax


    def _shift_sites(self):
        """
        Translate all coordinates such that the maximum of the
        maximum of the charge density is around the center
        of the cell.
        """


        # Translate atomic positions so zmax is in the center of the cell

        self.structure.translate_sites(indices=range(self.structure.num_sites), \
                    vector=[0,0,self.structure.lattice.c*(.5-self.zmax)], \
                                  frac_coords=False)
        zmax_index=int(self.zmax*self.parchg.dim[2])

        ngzf = self.parchg.dim[2]

        new_dat = []
        for i in range(self.parchg.dim[0]):
            new_dat.append([])
            for j in  range(self.parchg.dim[1]):
                dat_slice = self.chgdata[i][j]
                if self.zmax>.5:
                     new_dat[i].append(np.concatenate((dat_slice[zmax_index-int(ngzf/2):zmax_index],
                                     dat_slice[zmax_index:],
                                     dat_slice[:zmax_index-int(ngzf/2)])))
                else:
                     new_dat[i].append(np.concatenate((dat_slice[zmax_index+int(ngzf/2):],
                                     dat_slice[:zmax_index],
                                     dat_slice[zmax_index:zmax_index+int(ngzf/2)])))


        self.chgdata = new_dat
        self.zmax = self.structure.cart_coords[:, 2].max()/ \
                        self.structure.lattice.c

        self.structure.to('POSCAR','POSCAR_centered')

    def _get_stm_cell(self, z_below="", z_above="", r_val="", r_tol=""):
        """
        Calculate a single STM image cell from DFT-simulation.

        NOTE: Skew in the volumetric grid due to non-orthogonal lattice vectors
        in the imaging plane is NOT accounted for in the creation of the image_cell.
        Use get_stm_image (wrapper) to return an image that contains the proper skew!

        Args:
            z_below: (float) thickness or depth from top in Angstroms
            z_above: (float) distance above the surface to consider
            r_val: (float) isosurface charge density plane
            r_tol: (float) tolerance to consider while determining isosurface

        Returns:
            A numpy array (np.float64) of the simulated image tile.

        Credit: Chaitanya Kolluru
        """
        # Get volumetric slab corresponding to z_below and z_above parameters
        rhos, nzmax = self._get_stm_vol(z_below=z_below, z_above=z_above)

        if self._parameter_check_r(r_val=r_val, r_tol=r_tol, rhos=rhos):

            # If absdiff between charge density value and r_val not is not within r_tol, switch pixels to off (i.e. -9E9)
            rhos[abs(rhos - r_val) >= r_tol] = -9E9

            # Find all the pixel values for which entire charge density volume is swiched off
            mask = np.sum(rhos,axis=0) == np.shape(rhos)[0]*-9E9

            # Find all on pixels and flip bottom values to top
            temp = (rhos.copy() > 0)[::-1,...]

            # Create image by finding depth of first nonzero rhos pixel (argmax) and subtract from max-depth
            image = ((nzmax-1)-np.argmax(temp,axis=0)).astype(np.float64)

            # Fill in all 'off' pixels with min 'on' pixels value (minus 5) **** WHY IS THIS ???? ****
            image[mask==True] = np.min(image[mask==False]) - 5

            # Multiply indices by real-space height of volumetric slice
            image *= self.lwh[2]

            self.cell = image
            return image

    def simulate_image(self, sim_params=[]):
        """
        Calculate a full displayable STM image from DFT-simulation.

        Args:
            sim_params: (list) specifying the following simulation parameters
              - z_below: (float) thickness or depth from top in (Å)
              - z_above: (float) distance above the surface to consider (Å)
              - r_val: (float) isosurface charge density plane
              - r_tol: (float) tolerance to consider while determining isosurface
              - x_shear: (float) fractional amt shear in x (+ to the right)
              - y_shear: (float) fractional amt shear in y (+ up direction)
              - x_stretch: (float) fractional amt stretch (+) or compression (-) in x
              - y_stretch: (float) fractional amt stretch (+) or compression (-) in y
              - rotation: (float) image rotation angle (in degrees) (+ is CCW)
              - pix_size: (float) real-space pixel size (Å)
              - sigma: (float) standard deviation for gaussian kernel used in postprocessing
              - crop_height: (int) final (cropped) image height in pixels
              - crop_width: (int) final (cropped) image width in pixels

        Returns:
            A numpy array (np.float64) of the full simulated image.
        """
        # Enforce max stretch/squeeze and max/min shear value (both directions 0.30)
        sim_params[4] = 0 #sorted((-0.50, sim_params[4], 0.50))[1]
        sim_params[5] = 0 #sorted((-0.50, sim_params[5], 0.50))[1]
        sim_params[6] = 0 #sorted((-0.50, sim_params[6], 0.50))[1]
        sim_params[7] = 0 #sorted((-0.50, sim_params[7], 0.50))[1]

        # Clamps the given pixel size to a range between 0.05 and 0.40 (Å)
        sim_params[8] = sorted((-360, sim_params[8], 360))[1]

        # Clamps the given pixel size to a range between 0.05 and 0.40 (Å)
        sim_params[9] = sorted((0.05, sim_params[9], 0.40))[1]

        # Clamps the given sigma value to a range between 0 and 10
        sim_params[10] = 0 #sorted((0, sim_params[10], 10))[1]
        # sim_params[10] = 0 # In case we want to get rid of blur completely

        # Simulate the image cell
        self._get_stm_cell(z_below=sim_params[0], z_above=sim_params[1], r_val=sim_params[2], r_tol=sim_params[3])

        # Construct a larger cell by repeating image_cell in both directions, NREPS = 8
        img_tiled = np.tile(self.cell,(16,16))

        # Add shear to account for a non-orthogonal unit cell (if necessary)
        img_tiled = self._apply_lattice_shear(img_tiled)

        # Resize image based on pixel_size
        img_tiled = iop.apply_resize(img_tiled, np.array([int(a) for a in np.shape(img_tiled) * np.array(self.lwh[::-1][1::]) * (1./sim_params[9])]))

        # Rotate tiled image according to rotation value
        img_tiled = iop.apply_rotation(img_tiled, sim_params[8])

        # Apply any specified postprocessing shear
        img_tiled = iop.apply_shear(img_tiled, sim_params[4], sim_params[5])

        # Apply any specified postprocessing strech/squeeze
        img_tiled = iop.apply_stretch(img_tiled, sim_params[6], sim_params[7])

        if self._image_size_check(img=img_tiled):

            # Enforce odd sizes on width and length of cropped image (min_length = 25, max_length = current length)
            sim_params[-1] = sorted((37,int(2 * np.floor(sim_params[-1]/2) + 1),int(2 * np.floor((np.shape(img_tiled)[1]-2)/2) + 1)))[1]
            sim_params[-2] = sorted((37,int(2 * np.floor(sim_params[-2]/2) + 1),int(2 * np.floor((np.shape(img_tiled)[0]-2)/2) + 1)))[1]

            # Apply crop to image
            img_tiled = iop.apply_crop(img_tiled,sim_params[-1],sim_params[-2])

            # Apply gaussian blur
            image = iop.apply_blur(img_tiled, sigma=sim_params[10])

            # If sucessful, record parameters!
            self.sim_params = sim_params
            self.image = image
            return image, None

    def simulate_image_random(self, rotation="", pix_size="", crop_height="", crop_width=""):
        """
        Randomly simulate a valid image.

        Args:
            rotation: (float) image rotation angle (in degrees CCW)
            pix_size: (float) real-space edge length of a single pixel (in Å)
            crop_height: (int) final (cropped) image height in pixels
            crop_width: (int) final (cropped) image width in pixels

        Returns:
            A numpy array (np.float64) of the full simulated image.
        """
        sim_params = self._random_initialize(rotation=rotation, pix_size=pix_size, crop_height=crop_height, crop_width=crop_width)
        img_rand, __ = self.simulate_image(sim_params)
        return img_rand

    def _random_initialize(self, rotation="", pix_size="", crop_height="", crop_width=""):
        """
        Randomly select a set of parameters to simulate a valid image

        Args:
            rotation: (float) image rotation angle (in degrees CCW)
            pix_size: (float) real-space edge length of a single pixel (in Å)
            crop_height: (int) final (cropped) image height in pixels
            crop_width: (int) final (cropped) image width in pixels

        Returns:
            A numpy array of sim_params (tuple) specifying the necessary simulation parameters for 'simulate_image'
        """
        slab_thickness = self.structure.cart_coords[:, 2].max() - self.structure.cart_coords[:, 2].min()

        z_below = random.uniform(-1+1E-10,(0.5*slab_thickness)-1E-10)
        z_above = random.uniform(0,(1 - self.zmax) * self.structure.lattice.c)

        rhos, __ = self._get_stm_vol(z_below=z_below, z_above=z_above)

        r_val = random.uniform((rhos.max()/3)+1E-10, (rhos.max())-1E-10)
        r_tol = random.uniform(0.0001+1E-10,(0.999*r_val)-1E-10)

        x_shear  = random.uniform(-0.3,0.3)
        y_shear  = random.uniform(-0.3,0.3)
        x_stretch  = random.uniform(-0.3,0.3)
        y_stretch  = random.uniform(-0.3,0.3)

        sigma = random.uniform(0,4)

        if rotation != "":
            rotation = (rotation + random.choice(list(range(-360,360,90))) + random.uniform(-2,2))%360
        else:
            rotation = random.choice(list(range(-360,360,1)))

        if pix_size != "":
            pix_size = pix_size + random.uniform(-0.015,0.015)
        else:
            pix_size = random.uniform(0.05,0.40)

        if crop_width == "":
            crop_height = crop_width = int(2 * np.floor(random.uniform(33,121)/2) + 1)
        return [z_below, z_above, r_val, r_tol, x_shear, y_shear, x_stretch, y_stretch, rotation, pix_size, sigma, crop_height, crop_width]

    def _image_size_check(self, img):
        """
        """
        nrow, ncol = np.shape(img)
        if nrow > 35 and ncol > 35:
            pass
        else:
            raise Exception("ImageSizeError: {0} does not meet the minimum (37, 37) size requirement".format(np.shape(img)))
        return True

    def _parameter_check_z(self, z_below="", z_above=""):
        """
        Validate choice of depth parameters to ensure feasibility of STM image calculation

        Args:
            z_below: (float) thickness or depth from top in Angstroms
            z_above: (float) distance above the surface to consider

        Returns:
            A boolean (pass or fail)
        """
        slab_thickness = self.structure.cart_coords[:, 2].max() - self.structure.cart_coords[:, 2].min()

        # Check to make sure zbelow is within valid depth range (in Angstroms)
        if -1 <= z_below <= min([0.5*slab_thickness,2]):
            pass
        else:
            raise Exception("ParameterError: z_below = {0} outside valid range [{1} to {2:.5}]".format(z_below,-1,min([0.5*slab_thickness,2])))

        # Check to make sure z_above is within bounds of structure (within close proximity to surface, specifically)
        if 1 < z_above <= ((1 - self.zmax) * self.structure.lattice.c) - 2:
            pass
        else:
            raise Exception("ParameterError: z_above = {0} outside structure (must be <= {1:.5})".format(z_above,((1 - self.zmax) * self.structure.lattice.c)) - 2)
        return True

    def _parameter_check_r(self, r_val="", r_tol="", rhos=""):
        """
        Validate choice of charge density parameters to ensure feasibility of STM image calculation

        Args:
            r_val: (float) isosurface charge density plane
            r_tol: (float) tolerance to consider while determining isosurface
            rhos: (numpy array) the volumetric charge density slab

        Returns:
            A boolean (pass or fail)
        """
        # Check to make sure r_val is within valid isosurface (rmax/3 to rmax)
        if rhos.max()/3 < r_val <  rhos.max():
            pass
        else:
            raise Exception("ParameterError: r_val = {0} outside valid range ({1:.5} to {2:.5})".format(r_val,rhos.max()/3,rhos.max()))

        # Check to make sure r_tol is within proper tolerance (0.0001 to 0.999*r_val)
        if 0.0001 < r_tol <  0.999*r_val:
            pass
        else:
            raise Exception("ParameterError: r_tol = {0} outside valid range ({1:.5} to {2:.5})".format(r_tol,0.0001,0.999*r_val))
        return True

    def _apply_lattice_shear(self, img):
        """
        Apply centered shear of volumetric grid along lattice vectors.

        Args:
            img: (numpy array)

        Returns:
            A square numpy array (np.float64) of the sheared image.

        NOTE: This function has limited use cases. Add more for robust handling! Currently assumes "a"
              lattice vector in x-direction and "b" lattice vector resolves into x and y components.
        """
        # Get lattice vector a and b
        lva, lvb = self.parchg.structure.lattice.matrix[0], self.parchg.structure.lattice.matrix[1]

        # Use count of nonzero entries as method to detect vectors resolved along multiple directions
        if np.count_nonzero(lvb) == 1 and np.count_nonzero(lva) == 1:
            x_sh = 0
            y_sh = 0
        elif np.count_nonzero(lvb) == 2 and np.count_nonzero(lva) == 1:
            x_sh = -lvb[0]/np.linalg.norm(lva) # Shear b vector in the x direction
            y_sh = 0
        else:
            raise Exception('Shear detected, but not accounted for in image formed from volumetric grid!')
        return iop.apply_shear(img, x_sh, y_sh)

    def simulation_summary(self, iter="", verbose=False):
        """
        Write the parameter summary to screen (or config_file).

        Args:
            config_file: (string) name of the write file

        Returns:
            None
        """
        # Unpack parameters specific to simulation
        z_below, z_above, r_val, r_tol, x_shear, y_shear, x_stretch, y_stretch, rotation, pix_size, sigma, crop_height, crop_width = self.sim_params
        summary = """Iteration {0}:
        • (z_below, z_above) (Å)    :  {1}
        • (r_val, r_tol)            :  {2}
        • (x, y) shear (frac)       :  {3}
        • (x, y) stretch (frac)     :  {4}
        • sigma  (Gaussian blur)    :  {5}
        • rotation (deg CCW)        :  {6}
        • pix_size (Å)              :  {7}
        • img_size (pixels)         :  {8}""".format(iter,(z_below, z_above),(r_val, r_tol),(x_shear, y_shear),(x_stretch, y_stretch),sigma,rotation,pix_size,(crop_height,crop_width))
        if verbose:
            print(summary)
        else:
            return summary

    def display(self):
        """
        Display the simulated STM image.
        """
        plt.imshow(iop.scale_pixels(self.image,mode='grayscale'),cmap='hot')
        plt.axis('off')
        plt.show()

class Bicrystal(object):
    """
    This class contains two oriented supercell slabs prepared from structure queries to the Materials Project,
    as well as a calculator to fuse the slabs together into a bicrystal, simulate a convolution HAADF STEM image,
    and perform postprocessing imaging manipulations.
    """
    def __init__(self, config_file="", write_poscar=False, poscar_file=""):
        """
        Initialize a Bicrystal object with the path to a bicrystal slabs (json) file.

        Args:
            config_file: (string) path to a bicrystal slabs file
        """

        if poscar_file != "":
            poscar = Poscar.from_file(poscar_file)
            self.structure = poscar.structure
            pix_size = ""

        else:
            self.config_file = config_file
            slab = iop.open_construction_file(config_file)

            self.slab_1 = slab["slab_1"]
            self.slab_2 = slab["slab_2"]

            # Copy simulation folder from install directory to cwd
            try:
                shutil.copytree(os.path.dirname(__file__)+'/simulation', os.getcwd()+'/simulation')
            except:
                pass

            # Read constraints on dimensions for combined structure
            constraints = slab["constraints"]

            if "structure_file" not in slab:
                # Actual structure for the top grain (slab positioned into top of bicrystal cell)
                top_grain = TopGrain(self.slab_1["chemical_formula"], self.slab_1["space_group"],\
                                     self.slab_1['uvw_project'], self.slab_1['uvw_upward'],\
                                     self.slab_1['tilt_angle'], self.slab_1['max_dimension'],\
                                     self.slab_1['flip_species'])

                # Actual structure for the bottom grain (slab positioned into bottom of bicrystal cell)
                bot_grain = BottomGrain(self.slab_2["chemical_formula"], self.slab_2["space_group"],\
                                     self.slab_2['uvw_project'], self.slab_2['uvw_upward'],\
                                     self.slab_2['tilt_angle'], self.slab_2['max_dimension'],\
                                     self.slab_2['flip_species'])

                structure, top_grain_fit, bot_grain_fit, strain_info = self._get_base_structure(top_grain, bot_grain, constraints)

                self.top_grain = top_grain_fit
                self.bot_grain = bot_grain_fit
                self.structure = structure

            else:
                poscar = Poscar.from_file(slab["structure_file"])
                self.structure = poscar.structure

            try:
                pix_size = constraints['pixel_size']
            except:
                pix_size = ""

        # Image fields
        if pix_size != "":
            self.pix_size = pix_size
        else:
            self.pix_size = None

        # self.cell, __  = self._get_image_cell()
        # self.image = None
        # self.sim_params = None

        # self.lw = (self.structure.lattice.b/np.shape(self.cell)[1],
        #            self.structure.lattice.c/np.shape(self.cell)[0])

        self.cell = None
        self.image = None
        self.sim_params = None
        self.lw = None

        if write_poscar:
            self.write_poscar(strain_info=strain_info)

    def _get_base_structure(self, top_grain_structure, bot_grain_structure, constraints):
        """
        Take grains and fuse them into a simple bicrystal, subject to input constraints

        Args:
            top_grain_structure (): an ingrained.contruct.TopGrain() object
            bot_grain_structure (): an ingrained.contruct.BottomGrain() object
            constraints (dict): dictionary of geometric constraints for bicrystal construction

        Returns:
            The basic fused bicrystal (pymatgen structure) w/ corresponding ingrained.contruct.TopGrain()
            and ingrained.contruct.BottomGrain() objects after average expand/contract values applied
        """
        structure, top_grain, bot_grain, strain_info= self._fuse_grains(top_grain_structure, bot_grain_structure, constraints)
        structure = self._adjust_interface_width(structure=structure, interface_width_1=constraints['interface_1_width'], interface_width_2=constraints['interface_2_width'])
        structure = self._remove_interface_collisions(structure=structure, collision_removal_1=constraints['collision_removal'][0], collision_removal_2=constraints['collision_removal'][1])

        return structure, top_grain, bot_grain, strain_info

    def _fuse_grains(self, top_grain_structure, bot_grain_structure, constraints):
        """
        Take current grain structures and expand/contract in width/depth to minimize strain
        between grains and assign same width/depth computed as the average of the ideal
        expand/contract values for each grain.

        Args:
            top_grain_structure (): an ingrained.contruct.TopGrain() object
            bot_grain_structure (): an ingrained.contruct.BottomGrain() object
            constraints (dict): dictionary of geometric constraints for bicrystal construction

        Return:
            A bicrystal (pymatgen structure) from fusion of top and bottom grains
        """
        # Find integer expansions that minimize strain between lattices
        top_n_width, bot_n_width, tol_width = self._find_optimal_expansion(top_grain_structure.width, bot_grain_structure.width, \
                                                                      min_len=constraints['min_width'], max_len=constraints['max_width'])
        top_n_depth, bot_n_depth, tol_depth = self._find_optimal_expansion(top_grain_structure.depth, bot_grain_structure.depth, \
                                                                      min_len=constraints['min_depth'], max_len=constraints['max_depth'])

        # Apply appropriate integer expansions to grain structures
        top_grain_structure.structure.make_supercell([top_n_width,1,top_n_depth])
        bot_grain_structure.structure.make_supercell([bot_n_width,1,bot_n_depth])

        # Get expanded width/depth values of the grains
        widths = [top_grain_structure.structure.lattice.a,bot_grain_structure.structure.lattice.a]
        depths = [top_grain_structure.structure.lattice.c,bot_grain_structure.structure.lattice.c]

        # Use "average" width/depth of grains for bicrystal dimensions
        width_bc  = np.mean(widths)
        depth_bc  = np.mean(depths)
        height_bc = top_grain_structure.height * 2

        # Details on tension/compression
        strain_top_width = ((width_bc-widths[0])/widths[0])*100
        strain_bot_width = ((width_bc-widths[1])/widths[1])*100
        strain_top_depth = ((depth_bc-depths[0])/depths[0])*100
        strain_bot_depth = ((depth_bc-depths[1])/depths[1])*100

        # Get ASE Atoms object from pymatgen structures
        top_grain = AseAtomsAdaptor.get_atoms(top_grain_structure.structure)
        bot_grain = AseAtomsAdaptor.get_atoms(bot_grain_structure.structure)

        # Strain to coincidence
        top_grain.set_cell([width_bc,height_bc,depth_bc],scale_atoms=True)
        bot_grain.set_cell([width_bc,height_bc,depth_bc],scale_atoms=True)

        # Set bottom grain as the gb structure
        gb_full = bot_grain.copy()

        # Get chemical symbols and positions for top grain
        top_grain_sym = top_grain.get_chemical_symbols()
        top_grain_pos = top_grain.get_positions()

        # Insert top_grain info into gb structure (on top of the exisiting 'bottom' structure)
        for i in range(len(top_grain_sym)):
            gb_full.append(Atom(top_grain_sym[i],top_grain_pos[i]))

        # Get positions, cell information and shuffle axes
        gb_pos = gb_full.get_positions()
        gb_abc = gb_full.get_cell_lengths_and_angles()[0:3]
        gb_full.set_positions(np.vstack([np.vstack([gb_pos[:,2],gb_pos[:,0]]),gb_pos[:,1]]).T)
        gb_full.set_cell([gb_abc[2],gb_abc[0],gb_abc[1]])

        # Convert back to pymatgen structure and resolve any boundary conflicts
        gb_full_structure = AseAtomsAdaptor.get_structure(gb_full)
        gb_full_structure = self._enforce_pb(gb_full_structure)
        gb_full_structure = self._resolve_pb_conflicts(gb_full_structure)
        return gb_full_structure, top_grain_structure, bot_grain_structure, [strain_top_width,strain_top_depth,strain_bot_width,strain_bot_depth]

    def _adjust_interface_width(self, structure="", interface_width_1=0, interface_width_2=0):
        """
        Adjust spacing between both interfaces between grains, with option to remove collision conflicts

        Args:
            structure (pymatgen structure): bicrystal structure
            interface_width_1: (float) spacing b/w max pos of bottom grain and min pos of top grain (Å)
            interface_width_2: (float) spacing b/w max pos of top grain and min pos of bottom grain (Å)

        Return:
            A bicrystal (pymatgen structure) that reflects specified interface widths
        """
        # Get copy of current structure
        bc = structure.copy()
        if interface_width_1 != 0:
            # Update atom coordinate positions based on modified length in the c-direction from interface_width_1
            new_crds, new_spec = [], []
            for idx in range(len(bc)):
                cx,cy,cz = bc.cart_coords[idx]
                if bc.cart_coords[idx][2] >= bc.lattice.c/2:
                    new_crds.append([cx/bc.lattice.a,cy/bc.lattice.b,(cz+interface_width_1)/(bc.lattice.c+interface_width_1)])
                else:
                    new_crds.append([cx/bc.lattice.a,cy/bc.lattice.b,cz/(bc.lattice.c+interface_width_1)])
                new_spec.append(str(bc.species[idx]).split('Element')[0])

            # Create new 'Lattice' from parameters and a new pymatgen 'Structure' with updated positions
            lattice = Lattice.from_parameters(a=bc.lattice.a, b=bc.lattice.b, c=(bc.lattice.c+interface_width_1), alpha=90, beta=90, gamma=90)
            new_struct_1 = Structure(lattice, new_spec, new_crds)

            # Get copy of current structure with atom positions
            bc = new_struct_1.copy()

        if interface_width_2 != 0:
            crds = bc.cart_coords
            # Shift all coords up half distance of interface_width_2
            crds[:,2] = (crds[:,2]-np.min(crds[:,2])) + interface_width_2/2

            # Update atom coordinate positions based on modified length in the c-direction from interface_width_2
            new_crds, new_spec = [], []
            for idx in range(len(bc)):
                cx,cy,cz = crds[idx]
                new_crds.append([cx/bc.lattice.a,cy/bc.lattice.b,cz/(np.max(crds[:,2]) + interface_width_2/2)])
                new_spec.append(str(bc.species[idx]).split('Element')[0])

            # Create new 'Lattice' from parameters and a new pymatgen 'Structure' with updated positions
            lattice = Lattice.from_parameters(a=bc.lattice.a, b=bc.lattice.b, c=(np.max(crds[:,2]) + interface_width_2/2), alpha=90, beta=90, gamma=90)
            new_struct_2 = Structure(lattice, new_spec, new_crds)

            # Get copy of current structure with atom positions
            bc = new_struct_2.copy()
        return bc

    def _remove_interface_collisions(self, structure="", collision_removal_1=False, collision_removal_2=False, collision_distance=1):
        """
        Remove atoms at the interface that are within 'collision_distance' of another atom,
        starting removal with atoms at the bottom of the interface region

        Args:
            structure (pymatgen structure): bicrystal structure
            collision_removal_1 (boolean): remove atoms within 'collision_distance' in volume around interface_1
            collision_removal_2 (boolean): remove atoms within 'collision_distance' in volume around interface_2
            collision_distance (float): atoms less than or equal to this distance are considered collisions

        Return:
            A bicrystal (pymatgen structure) with collisions removed within the interface volumes
        """
        # Get copy of current structure
        bc = structure.copy()

        # Interface 1 collision removal
        if collision_removal_1:

            # Get all atomic coordinates
            pos = bc.cart_coords

            # Define region surrounding interface where we check for conflict (3Å region)
            iw_zone = ((bc.lattice.c/2)-1.5,(bc.lattice.c/2)+1.5)

            # Retrieve all coordinates belonging to the iw_zone
            interface_coords = pos[np.where((pos[:,2]>=iw_zone[0]) & (pos[:,2]<=iw_zone[1]))]

            # Ensure they are sorted such that lower coordinates are tested first
            interface_coords = interface_coords[interface_coords[:,2].argsort()]

            for coord in interface_coords:
                # Get array of distances between coord and coords of current structure
                dist_check = distance_matrix([coord], bc.cart_coords)[0]

                # If an atom is within collision_distance, remove!
                if np.sum(dist_check <= collision_distance) > 1:
                    indx = list(dist_check).index(0)
                    bc.remove_sites([indx])

        # Interface 2 collision removal
        if collision_removal_2:

            # Get all atomic coordinates
            pos = bc.cart_coords

            # Define region surrounding interface where we check for conflict (3Å region)
            iw_zone = (0,1.5)

            # Retrieve all coordinates belonging to the iw_zone
            interface_coords = pos[np.where((pos[:,2]>=iw_zone[0]) & (pos[:,2]<=iw_zone[1]))]

            # Ensure they are sorted such that lower coordinates are tested first
            interface_coords = interface_coords[interface_coords[:,2].argsort()]

            for coord in interface_coords:
                # Move coordinate across PBC to check for conflict
                coord[2] = coord[2]+bc.lattice.c

                # Get array of distances between coord and coords of current structure
                dist_check = distance_matrix([coord], bc.cart_coords)[0]

                # If an atom is within collision_distance, remove!
                if np.sum(dist_check <= collision_distance) > 1:
                    indx = list(dist_check).index(0)
                    bc.remove_sites([indx])
        return bc

    def _get_image_cell(self, defocus=1, interface_width=0, pix_size=0.15, view=False):
        """
        Calculate a single HAADF STEM image cell from bicrystal structure.

        Args:
            defocus: (float) controls degree to which edges blur in microscopy image (Å)
            interface_width: (float) spacing b/w max pos of bottom_grain and min pos of top grain (Å)
            pix_size: (float) real-space pixel size (Å)
            view: (boolean) option to display cell after simulation

        Returns:
            A numpy array (np.float64) of the simulated image tile.
            A pymatgen structure with interface and collision adjustments
        """

        # Get copy of current structure
        bc =self.structure.copy()

        # Apply additional interface width adjustments (on top of interface_1_width which is set during initialization)
        bc = self._adjust_interface_width(structure=bc, interface_width_1=interface_width)
        bc = self._remove_interface_collisions(structure=bc, collision_removal_1=True)

        # Use pixel_size to define shape of the output image
        pixx, pixy = np.round(np.array(bc.lattice.abc)/pix_size)[1::].astype(np.int)

        fmt = '% 4d', '% 8.4f', '% 9.4f', '% 9.4f', '% 4.2f', '% 4.3f'
        # Write input structure file required for Kirkland STEM simulation
        with open(os.getcwd()+'/simulation/SAMPLE.XYZ', "w") as sf:
            sf.write('Kirkland incostem input format\n')
            sf.write(" "*5+"".join(str(format(word, '8.4f')).ljust(10) for word in [bc.lattice.b,bc.lattice.c,bc.lattice.a])+"\n")
            coords_list = np.array(bc.cart_coords.tolist())
            save_coords = np.vstack([np.array(bc.atomic_numbers),coords_list[:,1],bc.lattice.c-coords_list[:,2],coords_list[:,0],np.ones(len(coords_list[:,0])),0.076*np.ones(len(coords_list[:,0]))]).T
            np.savetxt(sf,save_coords,fmt=fmt)
            # np.save(os.getcwd()+'/simulation/coords_list.npy',save_coords)
            sf.write(" "*2+"-1")

        ## Uncomment for DEBUGGING (prints xyz of imaged structure)
        #with open(os.getcwd()+'/simulation/demo.xyz', "w") as cf:
        #    cf.write(str(len(bc.cart_coords.tolist()))+"\n")
        #    cf.write("demo xyz structure file\n")
        #    for idx in range(len(bc.cart_coords.tolist())):
        #        atom_position = [bc.cart_coords.tolist()[idx][1],bc.lattice.c-bc.cart_coords.tolist()[idx][2],bc.cart_coords.tolist()[idx][0]]
        #        cf.write(str(bc.species[idx])+" "+"".join(str(format(word, '8.6f')).ljust(10) for word in [atom_position[0],atom_position[1],atom_position[2]])+"\n")

        # Write parameter file required for Kirkland STEM simulation
        with open(os.getcwd()+'/simulation/params.txt', "w") as pf:
            pf.write('SAMPLE.XYZ\n1 1 1\nSAMPLE.TIF\n'+str(pixx)+" "+str(pixy)+"\n")
            pf.write("200 0 0 0 30\n100 150\nEND\n"+str(defocus)+"\n0")

        # Simulate image with Kirkland incostem
        with cd(os.getcwd()+'/simulation'):
            subprocess.call("./incostem-osx", stdout=subprocess.PIPE)

        image = np.array(plt.imread(os.getcwd()+'/simulation/SAMPLE.TIF')).astype(np.float64)

        os.remove(os.getcwd()+'/simulation/SAMPLE.XYZ')
        os.remove(os.getcwd()+'/simulation/SAMPLE.TIF')
        os.remove(os.getcwd()+'/simulation/params.txt')

        if view:
            plt.imshow(image,cmap='hot'); plt.axis('off')
            plt.show()

        self.cell = image
        self.lw = (bc.lattice.b/np.shape(image)[1],bc.lattice.c/np.shape(image)[0])

        return image, bc

    def simulate_image(self, sim_params=[]):
        """
        Calculate a full displayable STM image from DFT-simulation.

        Args:
            sim_params: (list) specifying the following simulation parameters
              - pix_size: (float) real-space pixel size (Å)
              - interface_width: (float) spacing b/w max pos of bottom_grain and min pos of top grain (Å)
              - defocus: (float) controls degree to which edges blur in microscopy image (Å)
              - x_shear: (float) fractional amt shear in x (+ to the right)
              - y_shear: (float) fractional amt shear in y (+ up direction)
              - x_stretch: (float) fractional amt stretch (+) or compression (-) in x
              - y_stretch: (float) fractional amt stretch (+) or compression (-) in y
              - crop_height: (int) final (cropped) image height in pixels
              - crop_width: (int) final (cropped) image width in pixels

        Returns:
            A numpy array (np.float64) of the full simulated image.
        """
        # Clamps the given pixel size to a range
        if self.pix_size != None:
            sim_params[0] = sorted((0.945*self.pix_size, sim_params[0], 1.055*self.pix_size))[1]
        else:
            # sim_params[0] = sorted((0.1, sim_params[0], 0.4))[1]
            sim_params[0] = sorted((0.05, sim_params[0], 0.4))[1]

        # Clamp the interface_width to a range between -2 and 2 (Å)
        sim_params[1] = sorted((-2, sim_params[1], 2))[1]

        # Clamp the defocus to a range between 0.5 and 1.75 (Å)
        sim_params[2] = sorted((0.5, sim_params[2], 1.75))[1]

        # Enforce max stretch/squeeze and max/min shear value (both directions 0.20)
        sim_params[3] = sorted((-0.20, sim_params[3], 0.20))[1]
        sim_params[4] = sorted((-0.20, sim_params[4], 0.20))[1]
        sim_params[5] = sorted((-0.20, sim_params[5], 0.20))[1]
        sim_params[6] = sorted((-0.20, sim_params[6], 0.20))[1]

        # Simulate the image cell
        old_struct = self.structure.copy()
        # print(self.lw)

        img_cell, new_struct = self._get_image_cell(defocus=sim_params[2], interface_width=sim_params[1], pix_size=sim_params[0], view=False)

        # if new_struct == old_struct:
        #     print("No modifications were made to the structure!")
        # else:
        #     print("The structure was modified!")

        # print(self.lw)

        # Construct a larger cell by repeating image_cell in both directions
        img_tiled = np.tile(img_cell,(1,4))

        # Resize image based on pixel_size
        img_tiled = iop.apply_resize(img_tiled, np.array([int(a) for a in np.shape(img_tiled) * np.array(self.lw[::-1]) * (1./sim_params[0])]))

        # Apply any specified postprocessing shear
        img_tiled = iop.apply_shear(img_tiled, sim_params[3], sim_params[4])

        # Apply any specified postprocessing strech/squeeze
        img_tiled = iop.apply_stretch(img_tiled, sim_params[5], sim_params[6])

        # Enforce odd sizes on width and length of cropped image (min_length = 25, max_length = current length)
        sim_params[-1] = sorted((35,int(2 * np.floor(sim_params[-1]/2) + 1),int(2 * np.floor((np.shape(img_tiled)[1]-2)/2) + 1)))[1]
        sim_params[-2] = sorted((35,int(2 * np.floor(sim_params[-2]/2) + 1),int(2 * np.floor((np.shape(img_tiled)[0]-2)/2) + 1)))[1]

        # Apply crop to image
        image = iop.apply_crop(img_tiled,sim_params[-1],sim_params[-2])

        # If sucessful, record parameters!
        self.sim_params = sim_params
        self.image = image
        return image, new_struct

    def simulate_image_random(self, pix_size="", crop_height="", crop_width=""):
        """
        Randomly simulate a valid image.

        Args:
            pix_size: (float) real-space edge length of a single pixel (in Å)
            crop_height: (int) final (cropped) image height in pixels
            crop_width: (int) final (cropped) image width in pixels

        Returns:
            A numpy array (np.float64) of the full simulated image.
        """
        sim_params = self._random_initialize(pix_size=pix_size, crop_height=crop_height, crop_width=crop_width)
        img_rand, __ = self.simulate_image(sim_params)
        return img_rand

    def _random_initialize(self, pix_size="", crop_height="", crop_width=""):
        """
        Randomly select a set of parameters to simulate a valid image

        Args:
            pix_size: (float) real-space edge length of a single pixel (in Å)
            crop_height: (int) final (cropped) image height in pixels
            crop_width: (int) final (cropped) image width in pixels

        Returns:
            A numpy array of sim_params (tuple) specifying the necessary simulation parameters for 'simulate_image'
        """
        interface_width = random.uniform(-2+(1E-10),2-(1E-10))

        defocus = random.uniform(0.5+(1E-10),1.75-(1E-10))

        x_shear  = random.uniform(-0.1,0.1)
        y_shear  = random.uniform(-0.1,0.1)
        x_stretch  = random.uniform(-0.1,0.1)
        y_stretch  = random.uniform(-0.1,0.1)

        if pix_size != "":
            pix_size = pix_size + random.uniform(-0.015,0.015)
        else:
            pix_size = random.uniform(0.05,0.40)

        if crop_width == "":
            if crop_height != "":
                crop_width = int(random.uniform(0.80,1.2)*crop_height)
            else:
                crop_width = int(random.uniform(75,201))

        if crop_height == "":
            if crop_width != "":
                crop_height = int(random.uniform(0.80,1.2)*crop_width)
            else:
                crop_height = int(random.uniform(75,201))

        return [pix_size, interface_width, defocus, x_shear, y_shear, x_stretch, y_stretch, crop_height, crop_width]

    @staticmethod
    def _find_optimal_expansion(x, y, min_len=5, max_len=100):
        """
        Given two floats, x and y, return integers a and b such that
        |ax - by| ≈ 0 subject to min/max constraints

        Args:
            x: value 1
            y: value 2
            min_length: min value of an approximate multiple
            max_length: max value of an approximate multiple

        Returns:
            Smallest integers that best approximate the objective
        """
        alist = [x*i for i in range(1,100) if x*i < max_len and x*i > min_len]
        blist = [y*i for i in range(1,100) if y*i < max_len and y*i > min_len]
        dmat = distance.cdist(np.array(alist).reshape(-1,1),np.array(blist).reshape(-1,1),'euclidean')
        result = min((min((v, c) for c, v in enumerate(row)), r) for r, row in enumerate(dmat))
        return int(alist[result[1]]/x), int(blist[result[0][1]]/y), np.min(dmat)

    @staticmethod
    def _enforce_pb(pymatgen_structure):
        """
        Wrap atoms that have fixed coords outside cell back to their coord in cell.
        Notice this sometimes happens when using AseAtomsAdaptor.get_structure() to convert ASE to pymatgen

        Args:
            pymatgen_structure

        Returns:
            pymatgen_structure
        """
        s = pymatgen_structure.copy()
        for i in range(len(s.frac_coords)):
            s[i] = s[i].specie, [s[i].frac_coords[0]%1, s[i].frac_coords[1]%1, s[i].frac_coords[2]%1]
        return s

    @staticmethod
    def _resolve_pb_conflicts(pymatgen_structure):
        """
        If atoms near the 0 bound (width or depth only) interfere w/ atoms @ max bounds w/ PBC's, delete them!

        Args:
            pymatgen_structure

        Returns:
            pymatgen_structure
        """
        s = pymatgen_structure.copy()
        pos = s.cart_coords
        indel, i = [] , 0
        for crd in pos:
            if crd[0] < 0.5:
                test_crd = crd.copy()
                test_crd[0] += s.lattice.a
                if np.any((distance_matrix([test_crd], pos)[0]< 1) == True) == True and i not in indel:
                    indel.append(i)
            if crd[1] < 0.5:
                test_crd = crd.copy()
                test_crd[1] += s.lattice.b
                if np.any((distance_matrix([test_crd], pos)[0]< 1) == True) == True and i not in indel:
                    indel.append(i)
            i+=1
        s.remove_sites(indel)
        return s

    def simulation_summary(self, iter="", verbose=False):
        """
        Write the parameter summary to screen (or config_file).

        Args:
            config_file: (string) name of the write file

        Returns:
            None
        """
        # Unpack parameters specific to simulation
        pix_size, interface_width, defocus, x_shear, y_shear, x_stretch, y_stretch, crop_height, crop_width = self.sim_params
        summary = """Iteration {0}:
        • pix_size (Å)              :  {1}
        • interface width (Å)       :  {2}
        • defocus (Å)               :  {3}
        • (x, y) shear (frac)       :  {4}
        • (x, y) stretch (frac)     :  {5}
        • img_size (pixels)         :  {6}""".format(iter,pix_size, interface_width, defocus, (x_shear, y_shear), (x_stretch, y_stretch), (crop_height,crop_width))
        if verbose:
            print(summary)
        else:
            return summary

    def write_poscar(self, filename='bicrystal.POSCAR.vasp', strain_info=""):
        """
        Write the pymatgen structure to a POSCAR file.
        """
        self.structure.to(filename=filename)
        if strain_info != "":
            summary = "-"*30+"\n"+"Strain in top grain (%)\n  >> width (along b) : {:.3f}\n  >> depth (along a) : {:.3f}\nStrain in bottom grain (%)\n  >> width (along b) : {:.3f}\n  >> depth (along a) : {:.3f}\n".format(*strain_info)+"-"*30
            print(summary)
            print(summary,file=open("strain_info.txt", "w"))

    def display(self):
        """
        Display the simulated STM image.
        """
        plt.imshow(iop.scale_pixels(self.image,mode='grayscale'),cmap='gray')
        plt.axis('off')
        plt.show()
