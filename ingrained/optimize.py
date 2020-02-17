import os
import cv2
import pySPM
import random
import warnings
# import subprocess
import numpy as np
import dm3_lib as dm3lib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# from pymatgen.io.vasp import Poscar, Structure, Lattice

from scipy.optimize import minimize
from skimage.draw import polygon
from skimage.filters import rank
from skimage.morphology import disk
from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.exposure import equalize_adapthist
from skimage.feature import register_translation
from skimage.restoration import denoise_tv_chambolle

from .image_ops import *

# define Python user-defined exceptions
class Error(Exception):
   """Base class for other exceptions"""
   pass
class RegistrationSizeError(Error):
   """Raised when the input value is too large"""
   pass
class RegistrationSizeError(Error):
   """Raised when the input value is too large"""
   pass
class SimulationSizeError(Error):
   """Raised when the input value is too large"""
   pass

class CongruityBuilder(object):                   
    def __init__(self, bicrystal, dm3, objective='taxicab_vifp', optimizer='COBYLA'): 

        self.structure = bicrystal
        self.dm3_path  = dm3

    @staticmethod
    def _create_inset_window(fixed, coords=None ,moving=None, vertical_bias=0, horizontal_bias=0.125):
        """
        Create an inset window (turn off pixels outside window) inside a 'fixed' image by providing 
        a list of coordinate positions to assign to the window, or by a providing sliding window 
        'moving' image plus vertical and horizontal biases to create an inset window that contains 
        all translations of the 'moving' image within 'fixed'. 

        Args:
            fixed: A numpy array of the fixed image on which inset is created
            coords (optional): A list of tuples providing the coordinates of the window
            moving (optional): A numpy array of the moving image whose translations must be contained within fixed
            vertical_bias: A float (percentage) of the window to crop in the vertical direction
            horizontal_bias: A float (percentage) of the window to crop in the horizontal direction

        Returns:
            The fixed image (numpy array) with all pixels outside the windoe set to 0
        """
        rfix, cfix = np.shape(fixed)
        if moving is not None:
            rmov, cmov = np.shape(moving)
            # print(rmov, cmov)
            # Define boundaries based on input biases and ensure start<end
            rstt, rend = np.sort([int(np.ceil((rmov/2)+0.1))+int(horizontal_bias*(rmov/2)), \
                                  int(np.floor(rfix-(rmov/2)))-int(horizontal_bias*(rmov/2))])
            cstt, cend = np.sort([int(np.ceil((cmov/2)+0.1))+int(vertical_bias*(cmov/2)), \
                                  int(np.floor(cfix-(cmov/2)))-int(vertical_bias*(cmov/2))])
            # For the special case where start and end are the same:
            rstt, rend = ([rstt,rend],[rstt-1,rend+1])[rstt==rend]
            cstt, cend = ([cstt,cend],[cstt-1,cend+1])[cstt==cend]

            # Mask border around similarity_map to contain all possible translations of moving_downsample on fixed_downsample
            return cv2.copyMakeBorder(fixed[rstt:rend,cstt:cend],rstt,rfix-rend,cstt,cfix-cend,cv2.BORDER_CONSTANT,value=0)
        
        if coords is not None:
            # Create a mask and multiply with fixed
            mask = fixed.copy()
            mask[[xy[0] for xy in coords],[xy[1] for xy in coords]]= 100
            return (mask > 99) * fixed 

    @staticmethod
    def _windowed_histogram_similarity(fixed, moving):
        """
        For each pixel in the 'fixed', a histogram of the greyscale values in a region of the image 
        surrounding the pixel is computed and compared to a histogram of the greyscale values in the
        'moving' window to produce a similarity map, which indicates the level of similarity between 
        the 'moving' window centered on that pixel and the fixed image. 

        Adapted from: http://devdoc.net/python/scikit-image-doc-0.13.1/auto_examples/...
        features_detection/plot_windowed_histogram.htm
        
        Args:
            fixed: A numpy array of the fixed image on which the 'moving' image is slid across
            moving: A numpy array of the moving image slid across 'fixed' 

        Returns:
            A Chi squared similarity map (numpy array)
        """
        # Get row/col sizes for fixed (similarity map) and the image being shifted around
        row_fix, col_fix = np.shape(fixed)
        row_mov, col_mov = np.shape(moving)

        try: 
            if row_fix<row_mov or col_fix<col_mov:
                raise SimulationSizeError
        except:
            print("Error: Simulated image size exceeds its experimental target!")
            return None

        # Compute histogram for simulated image and normalize
        h_mov, _ = np.histogram(moving.flatten(), bins=16, range=(0, 16))
        h_mov = h_mov.astype(float) / np.sum(moving)

        # Compute normalized windowed histogram feature vector for each pixel (using a disk shaped mask)
        px_histograms = rank.windowed_histogram(img_as_ubyte(fixed), selem=disk(30), n_bins=h_mov.shape[0])

        # Reshape coin histogram to (1,1,N) for broadcast when we want to use it in
        # arithmetic operations with the windowed histograms from the image
        h_mov = h_mov.reshape((1, 1) + h_mov.shape)

        # Compute Chi squared distance metric: sum((X-Y)^2 / (X+Y));
        denom = px_histograms + h_mov
        denom[denom == 0] = np.infty
        frac = num = (px_histograms - h_mov) ** 2 / denom
        chi_sqr = 0.5 * np.sum(frac, axis=2)

        # Use a Chi squared similarity measure. 
        return 1 / (chi_sqr + 1.0e-4)

    @staticmethod
    def _taxicab_translation_distance(similarity_map,moving,fixed,max_iteration=10000):
        """
        Find the central coordinate which best fixes moving on fixed by minimizing 
        'taxicab' translation distance in an interative loop.
        
        Args:
            similariy_map: A numpy array indicating level of similarity b/w moving & fixed at each pixel
            fixed: A numpy array of the fixed image on which the 'moving' image is slid across
            moving: A numpy array of the moving image slid across 'fixed' 
            max_iteration: An int for the maximum iterations of the registration loop 

        Returns:
            A numpy array of the cropped 'fixed' that best matches 'moving'
            Central coordinate (pixel) of optimal position for 'moving' on 'fixed'
            Taxicab distance (int) between 'moving' and 'fixed'
        """
        shift_errors = 1e5*np.zeros(max_iteration)
        keep_indices = []
        i, shift = 0,-99

        while (i < max_iteration) and (np.sum(np.abs(shift)) > 0):

            # find indices of max similarity point
            current_best = np.unravel_index(similarity_map.argmax(), similarity_map.shape)
            experiment_patch = cutout_around_pixel(moving,fixed,current_best) 

            try: 
                if np.shape(moving) != np.shape(experiment_patch):
                    raise RegistrationSizeError
            except:
                print("Error: Registration requires image size agreement!")
                return None, None, None
    
            shift, error, diffphase = register_translation(moving,experiment_patch)   

            shift_errors[i] = np.sum(np.abs(shift))
            keep_indices.append(current_best)

            similarity_map[current_best] = 0

            if i == max_iteration-1:
                possible_idx = np.where(shift_errors == shift_errors.min())[0]
                current_best = keep_indices[possible_idx[np.argmin(np.abs((max_iteration/2)-possible_idx))]]# Find min closest to center pixel
                print('Warning: Proceed with caution! (SHIFT ERROR: {})'.format(int(np.sum(np.abs(shift)))))
            i += 1

        return experiment_patch, current_best, np.sum(np.abs(shift))

    def fit(self, interface_width=0, defocus=1.0, border_reduce=(0,0), pixel_size="", save_simulation = "", save_experiment = "", denoise=False):
        """
        A wrapper around '_taxicab_translation_distance' that allows for adjustments of structure
        and imaging parameters. Used in optimization loop to find set of parameters that yield
        best fit between simulation and experiment.
        
        Args:
            interface_width: A float that gives the spacing betweeen grains (relative to initial structure)
            defocus: A float used to specify the microsopy parameter in simulation
            border_reduce: A tuple that gives the extent of the vertical/horizontal border cropped during comparisons
            save_simulation: An path to the generated simulated image 
            save_simulation: An path to the appropriately cropped experimental image

        Returns:
            Both the 'fit' experimental and simulated image and the '_taxicab_translation_distance' between them
        """
        # Simulate an image using specified parameters
        simulated_raw = self.structure.convolution_HAADF(filename=save_simulation, dm3=self.dm3_path, \
                                                         pixel_size=pixel_size, interface_width=interface_width, \
                                                         defocus=defocus, border_reduce=border_reduce)

        # If simulation fails, None return for fit!
        if simulated_raw is None:
            return None, None, (None, None)

        # Read dm3 image from file
        dm3_image  = dm3lib.DM3(self.dm3_path).imagedata

        # Perform total-variation denoising on experimental image
        experiment_image  = denoise_tv_chambolle(dm3_image, weight=0.001, eps=0.005, n_iter_max=200)

        # Perform contrast limited adaptive histogram equalization for local contrast enhancement on simulation
        simulated_image   = equalize_adapthist(simulated_raw,clip_limit=0.5)

        # Quantize to 16 levels of greyscale and downsample (output image will have a 16-dim feature vec per pixel)        
        fixed_downsample  = custom_discretize(experiment_image,factor=4,mode="downsample")
        moving_downsample = custom_discretize(simulated_image,factor=4,mode="downsample")

        # Compute the similarity map for moving downsampled window across fixed downsampled image
        similarity_map = self._windowed_histogram_similarity(fixed_downsample,moving_downsample)
        
        # If simulation is too large, None return for fit!
        if similarity_map is None:
            return None, None, (None, None)

        # We are only interested in searching pixels towards the center of the image (with a slight horizontal bias)
        similarity_map  = self._create_inset_window(similarity_map,moving=moving_downsample,vertical_bias=0, horizontal_bias=0.25)

        # plt.imshow(similarity_map)
        # plt.show()

        # Find the optimal index on for moving image to be inset into the fixed image and return the associated cropped fixed image 
        _ , index_ds, shift_ds = self._taxicab_translation_distance(similarity_map,moving_downsample,fixed_downsample,max_iteration=np.sum(similarity_map>0))

        # If registration fails or is of low confidence (shift>0), None return for fit!
        if shift_ds is None or shift_ds>20:
            return None, None, (None, None)

        # Get upsampled pixels around 'index_ds' and check to see if same upsampled version is locked in place 
        upsampled_coordinates = get_rectangle_crds(np.array(index_ds)[0]*4-4, np.array(index_ds)[1]*4-4, (2*4)+1, (2*4)+1)
        
        upsampled_map = self._create_inset_window(np.ones(np.shape(experiment_image)),coords=upsampled_coordinates)

        _ , index_us, shift_us = self._taxicab_translation_distance(upsampled_map, \
                                                                    pixel_value_rescale(simulated_image,"uint4"), \
                                                                    pixel_value_rescale(experiment_image,"uint4"), \
                                                                    max_iteration=len(upsampled_coordinates))
        
        # If registration fails or is of low confidence (shift>4), None return for fit!
        if shift_us is None or shift_us>30:
            return None, None, (None, None)

        if denoise:
            # Get correspoinding cropped region of experimental image
            experiment_raw_image = cutout_around_pixel(simulated_image,experiment_image,index_us)
        else:
            # Get correspoinding cropped region of experimental image
            experiment_raw_image = cutout_around_pixel(simulated_image,dm3_image,index_us)

        # Write the initial simulated image to file
        if save_experiment:
            os.makedirs(os.path.dirname(save_experiment), exist_ok=True)
            experiment_raw_image = pixel_value_rescale(experiment_raw_image,dtype="uint8")
            cv2.imwrite(save_experiment, experiment_raw_image)

        return self.structure.haadf_image, experiment_raw_image, shift_us

    def taxicab_vifp_objective(self,x):
        interface_width, defocus, brdx, brdy, pixel_size = x
        print("Current Solution: \n>>> IW : {}\n>>> DF : {}\n>>> PX : {}\n>>> BX : {}\n>>> BY : {}".format(interface_width, defocus, pixel_size, brdx, brdy))
        simulated_raw, experiment_raw_image, shift_us = self.fit(interface_width=interface_width, defocus=defocus, border_reduce=(brdx,brdy), pixel_size=pixel_size, save_simulation = "", save_experiment = "",denoise=True)
        if simulated_raw is not None:
            match_vifp = score_vifp(simulated_raw,experiment_raw_image,sigma=2)
            fom = 0.1*(shift_us)+match_vifp
        else:
            fom = 9999
        print("🌀 FOM :",fom,"\n")
        return fom

    def find_correspondence(self,objective='taxicab_vifp', optimizer='COBYLA', initial_solution=[0,1.5,0.0,0.125,0.2], constraint_list=[]):

        def make_constraint_list(c):
            constraints = []
            for idx in range(len(c)):
                constraints.append({'type': 'ineq', 'fun': lambda t, idx=idx: t[idx] - c[idx][0]})
                constraints.append({'type': 'ineq', 'fun': lambda t, idx=idx: c[idx][1] - t[idx]})
            return constraints

        if optimizer == 'COBYLA':
            constraints = make_constraint_list(constraint_list)
            return minimize(self.taxicab_vifp_objective, initial_solution, method='COBYLA',tol=1E-6,options={'disp': True, 'rhobeg': 0.25, 'catol': 0.01}, constraints=constraints)

        if optimizer == 'Powell':
            return minimize(self.taxicab_vifp_objective, initial_solution, method='Powell',tol=1E-6,options={'disp': True})

class CongruityBuilderSTM(object):                   
    def __init__(self, partial_charge, sxm, restrict_bounds, objective='taxicab_vifp', optimizer='Powell'): 

        self.structure = partial_charge
        self.sxm_path = sxm
        self.restrict_bounds = restrict_bounds

    @staticmethod
    def _create_inset_window(fixed, coords=None ,moving=None, vertical_bias=0, horizontal_bias=0.125):
        """
        Create an inset window (turn off pixels outside window) inside a 'fixed' image by providing 
        a list of coordinate positions to assign to the window, or by a providing sliding window 
        'moving' image plus vertical and horizontal biases to create an inset window that contains 
        all translations of the 'moving' image within 'fixed'. 

        Args:
            fixed: A numpy array of the fixed image on which inset is created
            coords (optional): A list of tuples providing the coordinates of the window
            moving (optional): A numpy array of the moving image whose translations must be contained within fixed
            vertical_bias: A float (percentage) of the window to crop in the vertical direction
            horizontal_bias: A float (percentage) of the window to crop in the horizontal direction

        Returns:
            The fixed image (numpy array) with all pixels outside the windoe set to 0
        """
        rfix, cfix = np.shape(fixed)
        if moving is not None:
            rmov, cmov = np.shape(moving)
            # print(rmov, cmov)
            # Define boundaries based on input biases and ensure start<end
            rstt, rend = np.sort([int(np.ceil((rmov/2)+0.1))+int(horizontal_bias*(rmov/2)), \
                                  int(np.floor(rfix-(rmov/2)))-int(horizontal_bias*(rmov/2))])
            cstt, cend = np.sort([int(np.ceil((cmov/2)+0.1))+int(vertical_bias*(cmov/2)), \
                                  int(np.floor(cfix-(cmov/2)))-int(vertical_bias*(cmov/2))])
            # For the special case where start and end are the same:
            rstt, rend = ([rstt,rend],[rstt-1,rend+1])[rstt==rend]
            cstt, cend = ([cstt,cend],[cstt-1,cend+1])[cstt==cend]

            # Mask border around similarity_map to contain all possible translations of moving_downsample on fixed_downsample
            return cv2.copyMakeBorder(fixed[rstt:rend,cstt:cend],rstt,rfix-rend,cstt,cfix-cend,cv2.BORDER_CONSTANT,value=0)
        
        if coords is not None:
            # Create a mask and multiply with fixed
            mask = fixed.copy()
            mask[[xy[0] for xy in coords],[xy[1] for xy in coords]]= 100
            return (mask > 99) * fixed 

    @staticmethod
    def _windowed_histogram_similarity(fixed, moving):
        """
        For each pixel in the 'fixed', a histogram of the greyscale values in a region of the image 
        surrounding the pixel is computed and compared to a histogram of the greyscale values in the
        'moving' window to produce a similarity map, which indicates the level of similarity between 
        the 'moving' window centered on that pixel and the fixed image. 

        Adapted from: http://devdoc.net/python/scikit-image-doc-0.13.1/auto_examples/...
        features_detection/plot_windowed_histogram.htm
        
        Args:
            fixed: A numpy array of the fixed image on which the 'moving' image is slid across
            moving: A numpy array of the moving image slid across 'fixed' 

        Returns:
            A Chi squared similarity map (numpy array)
        """
        # Get row/col sizes for fixed (similarity map) and the image being shifted around
        row_fix, col_fix = np.shape(fixed)
        row_mov, col_mov = np.shape(moving)

        try: 
            if row_fix<row_mov or col_fix<col_mov:
                raise SimulationSizeError
        except:
            print("Error: Simulated image size exceeds its experimental target!")
            return None

        # Compute histogram for simulated image and normalize
        h_mov, _ = np.histogram(moving.flatten(), bins=16, range=(0, 16))
        h_mov = h_mov.astype(float) / np.sum(moving)

        # Compute normalized windowed histogram feature vector for each pixel (using a disk shaped mask)
        px_histograms = rank.windowed_histogram(img_as_ubyte(fixed), selem=disk(30), n_bins=h_mov.shape[0])

        # Reshape coin histogram to (1,1,N) for broadcast when we want to use it in
        # arithmetic operations with the windowed histograms from the image
        h_mov = h_mov.reshape((1, 1) + h_mov.shape)

        # Compute Chi squared distance metric: sum((X-Y)^2 / (X+Y));
        denom = px_histograms + h_mov
        denom[denom == 0] = np.infty
        frac = num = (px_histograms - h_mov) ** 2 / denom
        chi_sqr = 0.5 * np.sum(frac, axis=2)

        # Use a Chi squared similarity measure. 
        return 1 / (chi_sqr + 1.0e-4)

    @staticmethod
    def _taxicab_translation_distance(similarity_map,moving,fixed,max_iteration=10000):
        """
        Find the central coordinate which best fixes moving on fixed by minimizing 
        'taxicab' translation distance in an interative loop.
        
        Args:
            similariy_map: A numpy array indicating level of similarity b/w moving & fixed at each pixel
            fixed: A numpy array of the fixed image on which the 'moving' image is slid across
            moving: A numpy array of the moving image slid across 'fixed' 
            max_iteration: An int for the maximum iterations of the registration loop 

        Returns:
            A numpy array of the cropped 'fixed' that best matches 'moving'
            Central coordinate (pixel) of optimal position for 'moving' on 'fixed'
            Taxicab distance (int) between 'moving' and 'fixed'
        """
        shift_errors = 1e5*np.zeros(max_iteration)
        keep_indices = []
        i, shift = 0,-99

        while (i < max_iteration) and (np.sum(np.abs(shift)) > 0):

            # find indices of max similarity point
            current_best = np.unravel_index(similarity_map.argmax(), similarity_map.shape)
            experiment_patch = cutout_around_pixel(moving,fixed,current_best) 

            try: 
                if np.shape(moving) != np.shape(experiment_patch):
                    raise RegistrationSizeError
            except:
                print("Error: Registration requires image size agreement!")
                return None, None, None
    
            shift, error, diffphase = register_translation(moving,experiment_patch)   

            shift_errors[i] = np.sum(np.abs(shift))
            keep_indices.append(current_best)

            similarity_map[current_best] = 0

            if i == max_iteration-1:
                possible_idx = np.where(shift_errors == shift_errors.min())[0]
                current_best = keep_indices[possible_idx[np.argmin(np.abs((max_iteration/2)-possible_idx))]]# Find min closest to center pixel
                print('Warning: Proceed with caution! (SHIFT ERROR: {})'.format(int(np.sum(np.abs(shift)))))
            i += 1

        return experiment_patch, current_best, np.sum(np.abs(shift))

    def fit(self, zthick="", ztol="", rho0="", rho_tol="", pixel_size="", rotation_angle="", save_simulation = "", save_experiment = "", display=False):
        """
        A wrapper around '_taxicab_translation_distance' that allows for adjustments of structure
        and imaging parameters. Used in optimization loop to find set of parameters that yield
        best fit between simulation and experiment.
        
        Args:
            interface_width: A float that gives the spacing betweeen grains (relative to initial structure)
            defocus: A float used to specify the microsopy parameter in simulation
            border_reduce: A tuple that gives the extent of the vertical/horizontal border cropped during comparisons
            save_simulation: An path to the generated simulated image 
            save_simulation: An path to the appropriately cropped experimental image

        Returns:
            Both the 'fit' experimental and simulated image and the '_taxicab_translation_distance' between them
        """
        # Simulate an image using specified parameters

        simulated_raw = self.structure.stm(filename=save_simulation, dm3=self.sxm_path, \
                                            pixel_size=pixel_size, rotation_angle=rotation_angle, \
                                            zthick=zthick, ztol=ztol, rho0=rho0, rho_tol=rho_tol)

        # If simulation fails, None return for fit!
        if simulated_raw is None:
            return None, None, (None, None)

        # Read sxm image from file
        sxm_image  = pySPM.SXM(self.sxm_path).get_channel('Z', direction='both',corr="slope").pixels

        # Restrict search area to clean part of sample
        # rrow_start, rrow_end = (0.10,0.65)
        # rcol_start, rcol_end = (0.4,1.0)

        (rrow_start, rrow_end) , (rcol_start, rcol_end) = self.restrict_bounds

        experiment_image = sxm_image[int(np.floor(np.shape(sxm_image)[0]*rrow_start)):int(np.floor(np.shape(sxm_image)[0]*rrow_end)),\
                                     int(np.floor(np.shape(sxm_image)[1]*rcol_start)):int(np.floor(np.shape(sxm_image)[1]*rcol_end))]

        # Works well!
        # experiment_image = experiment_image[int(np.floor(np.shape(experiment_image)[0]*0.10)):int(np.floor(np.shape(experiment_image)[0]*0.65)),int(np.floor(np.shape(experiment_image)[1]*0.40))::]

        # Perform contrast limited adaptive histogram equalization for local contrast enhancement on simulation
        # simulated_image   = equalize_adapthist(simulated_raw,clip_limit=0.5)
        simulated_image   = simulated_raw

        # Quantize to 16 levels of greyscale and downsample (output image will have a 16-dim feature vec per pixel)        
        fixed_downsample  = custom_discretize(experiment_image,factor=4,mode="downsample")
        moving_downsample = custom_discretize(simulated_image,factor=4,mode="downsample")

        # Compute the similarity map for moving downsampled window across fixed downsampled image
        similarity_map = self._windowed_histogram_similarity(fixed_downsample,moving_downsample)
        
        # If simulation is too large, None return for fit!
        if similarity_map is None:
            return None, None, (None, None)

        # We are only interested in searching pixels towards the center of the image (with a slight horizontal bias)
        similarity_map  = self._create_inset_window(similarity_map,moving=moving_downsample,vertical_bias=0, horizontal_bias=0)

        # Find the optimal index on for moving image to be inset into the fixed image and return the associated cropped fixed image 
        _ , index_ds, shift_ds = self._taxicab_translation_distance(similarity_map,moving_downsample,fixed_downsample,max_iteration=np.sum(similarity_map>0))

        # If registration fails or is of low confidence (shift>0), None return for fit!
        if shift_ds is None or shift_ds>20:
            return None, None, (None, None)

        # Get upsampled pixels around 'index_ds' and check to see if same upsampled version is locked in place 
        upsampled_coordinates = get_rectangle_crds(np.array(index_ds)[0]*4-4, np.array(index_ds)[1]*4-4, (2*4)+1, (2*4)+1)
        
        upsampled_map = self._create_inset_window(np.ones(np.shape(experiment_image)),coords=upsampled_coordinates)

        _ , index_us, shift_us = self._taxicab_translation_distance(upsampled_map, \
                                                                    pixel_value_rescale(simulated_image,"uint4"), \
                                                                    pixel_value_rescale(experiment_image,"uint4"), \
                                                                    max_iteration=len(upsampled_coordinates))
        
        if display:
            insert_image_patch_STM(simulated_image,experiment_image,upsampled_map,index_us,save_simulation.split("/")[0])

        # If registration fails or is of low confidence (shift>4), None return for fit!
        if shift_us is None or shift_us>30:
            return None, None, (None, None)

        experiment_raw_image = cutout_around_pixel(simulated_image,experiment_image,index_us)

        # Write the initial simulated image to file
        if save_experiment:
            os.makedirs(os.path.dirname(save_experiment), exist_ok=True)
            experiment_raw_image = pixel_value_rescale(experiment_raw_image,dtype="uint8")
            cv2.imwrite(save_experiment, experiment_raw_image)

        plt.imshow(self.structure.stm_image); 
        plt.show()

        return self.structure.stm_image, experiment_raw_image, shift_us

    def taxicab_vifp_objective(self,x):
        zval,ztol,rval,rtol,rang,pixs = x
        print("Current Solution: \n>>> ZThk : {}\n>>> ZTol : {}\n>>> RVal : {}\n>>> RTol : {}\n>>> Ang  : {}\n>>> Pix  : {}".format(zval,ztol,rval,rtol,rang,pixs))
        rval = float(rval)/1000
        rtol = float(rtol)/1000
        simulated_raw, experiment_raw_image, shift_us = self.fit(zthick=zval, ztol=ztol, rho0=rval, rho_tol=rtol, pixel_size=pixs, rotation_angle=rang, save_simulation = "", save_experiment = "", display=False)
        if simulated_raw is not None:
            match_vifp = score_vifp(simulated_raw,experiment_raw_image,sigma=2)
            fom = 0.1*(shift_us)+match_vifp
        else:
            fom = 9999
        print("🌀 FOM :",fom,"\n")
        return fom

    def find_correspondence(self,objective='taxicab_vifp', optimizer='COBYLA', initial_solution="", constraint_list=[]):

        def make_constraint_list(c):
            constraints = []
            for idx in range(len(c)):
                constraints.append({'type': 'ineq', 'fun': lambda t, idx=idx: t[idx] - c[idx][0]})
                constraints.append({'type': 'ineq', 'fun': lambda t, idx=idx: c[idx][1] - t[idx]})
            return constraints

        if optimizer == 'COBYLA':
            constraints = make_constraint_list(constraint_list)
            print(constraints)
            return minimize(self.taxicab_vifp_objective, initial_solution, method='COBYLA',tol=1E-6,options={'disp': True, 'rhobeg': 0.25, 'catol': 0.0002}, constraints=constraints)

        if optimizer == 'Powell':
            return minimize(self.taxicab_vifp_objective, initial_solution, method='Powell',tol=1E-6,options={'disp': True})
