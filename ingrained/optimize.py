import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import rank
from skimage.morphology import disk
from skimage.registration import phase_cross_correlation

from scipy.optimize import minimize

import ingrained.image_ops as iop

class CongruityBuilder(object):      
    """
    Find optimal correspondence "congruity" between a simulated and experimental image.
    Posed as a “jigsaw puzzle” problem where the experimental image is fixed and the goal 
    of optimization is to find a set of simulation parameters, 𝜃, that produce an image 
    that can be arranged in such a way inside the experimental image that minimizes: 
    
    𝐽(𝜃)= 𝛼*d_TC(𝜃) + 𝛽*d_𝑆𝑆𝐼𝑀(𝜃)  
    
    - where d_TC(𝜃) is the taxicab distance required for optimal cross-correlation-based 
      registration after upsampling (enforces consistency in the pattern across boundaries)

    - where d_SSIM(𝜃) is the Structural Similarity Index Measure, which quantifies the visual 
      similarity between the simulation and experiment patch (enforces visual consistency in 
      the content within the patches)

    - 𝛼, 𝛽 are weights to chosen to balance importance of the criteria

    """
    def __init__(self, sim_obj="", exp_img="", iter=0): 
        """
        Initialize a CongruityBuilder object with a ingrained.structure and experimental image.
                 
        Args:
            sim_obj: (ingrained.structure) one of Bicrystal or PartialCharge stuctures
            exp_img: (np.array) the experimental image (consider preprocessing to enhance optimization solution)
        """  
        self.sim_obj = sim_obj
        self.exp_img = exp_img
        self.iter    = iter
        
    def fit(self, sim_params = [], display=False, bias_x=0.0, bias_y=0.0):
        """
        Find optimal correspondence between simulation and experiment for the specified parameters.

        Args:
            sim_params: (list) parameters required for the 'simulate_image' method in the sim_obj
            display: (string) plt.show() for intermediate/final fit results 
            bias_x: (float) reduce search area in the x-direction by a fraction
            bias_y: (float) reduce search area in the y-direction by a fraction

        Returns:
            Both 'fit' experimental and simulated image with the translation_distance between them and 
            coodinates of the center pixel in the experiment
        """ 
        # Simulate an image using specified parameters
        sim_img, sim_struct = self.sim_obj.simulate_image(sim_params=sim_params)
        
        # Display simulated image
        # self.sim_obj.display()
        
        # Read experimental image
        exp_img = self.exp_img.copy()
        
        # Downsample (with quantization) both simulated and experimental images
        # ds_by_factor = 2
        ds_by_factor = 4
        ds_sim_img = iop.apply_quantize_downsample(sim_img, factor=ds_by_factor)
        ds_exp_img = iop.apply_quantize_downsample(exp_img, factor=ds_by_factor)
        
        # Compute the Chi-squared (correlation) map for moving the downsampled simulation across downsampled experiment
        similarity_map = self.windowed_histogram_similarity(fixed=ds_exp_img, moving=ds_sim_img, bias_x=bias_x, bias_y=bias_y)
        # plt.imshow(similarity_map,cmap='hot'); plt.show()

        # Find pixel of fixed image where moving image can be positioned and not need further translation to improve similarity score
        ds_exp_patch, ds_stable_idxs, ds_shift_score = self.stabilize_inset_map(fixed=ds_exp_img, moving=ds_sim_img, similarity_map=similarity_map)
        
        # Quantizate both original simulated and experimental images (without downsampling, i.e. factor = 1)
        qt_sim_img = iop.apply_quantize_downsample(sim_img, factor=1)
        qt_exp_img = iop.apply_quantize_downsample(exp_img, factor=1)
        
        # Get upsampled idxs around 'ds_stable_idxs' 
        us_coords = iop.pixels_within_rectangle(np.array(ds_stable_idxs)[0]*ds_by_factor-ds_by_factor, 
                                                np.array(ds_stable_idxs)[1]*ds_by_factor-ds_by_factor, 
                                                (2*ds_by_factor)+1, (2*ds_by_factor)+1)
        
        # Check if moving image is locked in place over upsampled (original spatial resolution) images
        us_exp_patch, us_stable_idxs, us_shift_score = self.stabilize_inset_coords(fixed=qt_exp_img, moving=qt_sim_img, critical_idxs=us_coords)
        
        # Get the shape of the moving image (size used as reference to extract exp_patch)
        nrows, ncols = np.shape(sim_img)
        
        # Get best fit experiment patch after search (us_exp_patch is quantized, we want patch from original image w/o quantization!)
        exp_patch = exp_img[int(us_stable_idxs[0]-(nrows-1)/2):int(us_stable_idxs[0]+(nrows-1)/2)+1,\
                            int(us_stable_idxs[1]-(ncols-1)/2):int(us_stable_idxs[1]+(ncols-1)/2)+1]
        
        # Convert to grayscale for viewing
        gs_exp_img = iop.scale_pixels(exp_patch,mode='grayscale')
        gs_sim_img = iop.scale_pixels(sim_img,mode='grayscale')
    
        if display:
            # Display matching pair after downsample stabilized
            plt.imshow(np.hstack([ds_exp_patch,15*np.ones((np.shape(ds_exp_patch)[0],5)),ds_sim_img]),cmap='hot'); plt.axis('off');
            plt.show();

            # Display matching pair after upsample stabilized
            plt.imshow(np.hstack([us_exp_patch,15*np.ones((np.shape(us_exp_patch)[0],5)),qt_sim_img]),cmap='hot'); plt.axis('off');
            plt.show()

            # Display matching pair (native resolution) stabilized
            plt.imshow(np.hstack([gs_exp_img,255*np.ones((np.shape(gs_exp_img)[0],5)),gs_sim_img]),cmap='hot'); plt.axis('off');
            plt.show()
        return sim_img, sim_struct, exp_patch, us_shift_score, us_stable_idxs
    
    def fit_gb(self, sim_params = [], display=False, bias_x=0.0, bias_y=0.15):
        """
        Find optimal correspondence between simulation and experiment for the specified parameters.
        Bias added to search region which promotes searching closer to center of image (vertically)

        Args:
            sim_params: (list) parameters required for the 'simulate_image' method in the sim_obj
            bias_x: (float) reduce search area in the x-direction by a fraction
            bias_y: (float) reduce search area in the y-direction by a fraction

        Returns:
            Both 'fit' experimental and simulated image with the translation_distance between them and 
            coodinates of the center pixel in the experiment
        """ 
        return self.fit(sim_params=sim_params, display=False, bias_x=bias_x, bias_y=bias_y)

    def windowed_histogram_similarity(self, fixed="", moving="", bias_x=0.0, bias_y=0.0):
        """
        Compute the Chi squared distance metric between the moving image across all pixels in the fixed image,
        in an attempt to locate the moving image (or highly similar regions) within the original fixed image.
        
        Args:
            fixed: (np.array) experimental image 
            moving: (np.array) a simulated image patch (<= size of fixed image)
            bias_x: (float) reduce search area in the x-direction by a fraction
            bias_y: (float) reduce search area in the y-direction by a fraction
            
        Returns:
            A similarity map indicating regions of high correlation (reciprocal of Chi-squared similarity) between moving and fixed.
        """
        # Compute histogram for simulated (moving) image, and normalize
        moving_hist, _ = np.histogram(moving.flatten(), bins=16, range=(0, 16))
        moving_hist = moving_hist.astype(float) / np.sum(moving_hist)
                
        # Find appropriate size for a disk shaped mask that will define the shape of the sliding window
        radius = int((2/3) * np.max(np.shape(moving)))
        selem  = disk(radius) # A disk is (2 * radius + 1) wide 
           
        # Compute normalized windowed histogram feature vector for each pixel in the fixed image
        px_histograms = rank.windowed_histogram(fixed, selem=selem, n_bins=moving_hist.shape[0])

        # Reshape moving histogram to (1,1,N) for broadcast when we want to use it in
        # arithmetic operations with the windowed histograms from the image
        moving_hist = moving_hist.reshape((1, 1) + moving_hist.shape)
        
        # Compute Chi-squared distance metric: sum((X-Y)^2 / (X+Y));
        denom = px_histograms + moving_hist
        denom[denom == 0] = np.infty
        frac = num = (px_histograms - moving_hist) ** 2 / denom
        chi_sqr = 0.5 * np.sum(frac, axis=2)

        # Use reciprocal of Chi-squared similarity measure to create a full similarity map
        full_map = 1 / (chi_sqr + 1.0e-4)
        similarity_map = full_map.copy()
        
        # Get half length/width + 1 of moving image and use to define border size on fixed (where no sliding window comparisons permitted)
        pix_row = int(np.ceil(np.shape(moving)[0]/2) + 1)
        pix_col = int(np.ceil(np.shape(moving)[1]/2) + 1)
  
        # Bias values can further restrict area where comparisons are made
        pix_row += int(((np.shape(similarity_map)[0] - 2*pix_row) * bias_y)/2)
        pix_col += int(((np.shape(similarity_map)[1] - 2*pix_col) * bias_x)/2)
        
        # Construct the final correlation map 
        brd_col = np.zeros((pix_row,np.shape(similarity_map)[1]))
        brd_row = np.zeros((np.shape(similarity_map)[0],pix_col))
        
        similarity_map = np.vstack([brd_col,similarity_map[pix_row:-pix_row,::],brd_col])
        similarity_map = np.hstack([brd_row,similarity_map[::,pix_col:-pix_col],brd_row])
        return similarity_map
    
    def stabilize_inset_map(self, fixed="", moving="", similarity_map=""):
        """
        Find position (center pixel) on the fixed image, where the moving image can be placed without 
        further translation (or with minimal translation) to improve registration. The similarity map 
        defines the order that the center pixels are tested.
        
        Args:
            fixed: (np.array) experimental image 
            moving: (np.array) a simulated image patch (<= size of fixed image)
            similarity_map: (np.array) map of pixels measuring similarity between moving and fixed at a each center pixel
            
        Returns:
            The stable center pixel and the correponding experimental patch (same size as moving image), 
            as well as the absolute value of the shift vector (in pixels) required to register the images. 
            A stable return is currently defined as having a shift error of zero.  
        """
        # Get the shape of the moving image
        nrows, ncols = np.shape(moving)

        # Find the index (row, col) with the current highest similarity
        current_idx = np.unravel_index(similarity_map.argmax(), similarity_map.shape)

        # Keep track of the current best index and shift score 
        current_best = None

        for i in range(np.sum(similarity_map > 0)):

            # Get patch of fixed image that contains current_idx as the center pixel
            fixed_patch = fixed[int(current_idx[0]-(nrows-1)/2):int(current_idx[0]+(nrows-1)/2)+1,\
                                int(current_idx[1]-(ncols-1)/2):int(current_idx[1]+(ncols-1)/2)+1]

            # Find shift required for maximum cross correlation
            shift, __, __ = phase_cross_correlation(moving,fixed_patch)  

            # Calculate shift score 
            shift_score = np.sum(np.abs(shift))

            # Check to see if shift score improved, if so, record the index!
            if current_best == None or shift_score < current_best [2]:
                current_best = [current_idx[0],current_idx[1],shift_score]

            # If moving image cannot be further shifted to improve similarity with fixed, consider STABILIZED and break!
            if shift_score == 0:
                break

            # If keep going, reset the current_idx to "0" (indicates that it has been tested)
            similarity_map[current_idx] = 0

            # Find next best option
            current_idx = np.unravel_index(similarity_map.argmax(), similarity_map.shape)

        if shift_score != 0:
            print('Warning: Proceed with caution! (SHIFT ERROR: {})'.format(int(current_best[2])))

        # Get best fit patch after search
        fixed_patch = fixed[int(current_best[0]-(nrows-1)/2):int(current_best[0]+(nrows-1)/2)+1,\
                            int(current_best[1]-(ncols-1)/2):int(current_best[1]+(ncols-1)/2)+1]
        return fixed_patch, tuple(current_best[0:2]), int(current_best[2])

    def stabilize_inset_coords(self, fixed="", moving="", critical_idxs=""):
        """
        Find position (center pixel) on the fixed image, where the moving image can be placed without 
        further translation (or with minimal translation) to improve registration. The critical indices 
        provided define the order that the center pixels are tested.
        
        Args:
            fixed: (np.array) experimental image 
            moving: (np.array) a simulated image patch (<= size of fixed image)
            critical_idxs: (list) critical indices to test for stability 
            
        Returns:
            The stable center pixel and the correponding experimental patch (same size as moving image), 
            as well as the absolute value of the shift vector (in pixels) required to register the images. 
            A stable return is currently defined as having a shift error of zero.  
        """
        # Get the shape of the moving image
        nrows, ncols = np.shape(moving)

        # Keep track of the current best index and shift score 
        current_best = None

        for current_idx in critical_idxs:

            # Get patch of fixed image that contains current_idx as the center pixel
            fixed_patch = fixed[int(current_idx[0]-(nrows-1)/2):int(current_idx[0]+(nrows-1)/2)+1,\
                                int(current_idx[1]-(ncols-1)/2):int(current_idx[1]+(ncols-1)/2)+1]

            # Find shift required for maximum cross correlation
            shift, __, __ = phase_cross_correlation(moving,fixed_patch)  

            # Calculate shift score 
            shift_score = np.sum(np.abs(shift))

            # Check to see if shift score improved, if so, record the index!
            if current_best == None or shift_score < current_best [2]:
                current_best = [current_idx[0],current_idx[1],shift_score]

            # If moving image cannot be further shifted to improve similarity with fixed, consider STABILIZED and break!
            if shift_score == 0:
                break

        if shift_score != 0:
            print('Warning: Proceed with caution! (SHIFT ERROR: {})'.format(int(current_best[2])))

        # Get best fit patch after search
        fixed_patch = fixed[int(current_best[0]-(nrows-1)/2):int(current_best[0]+(nrows-1)/2)+1,\
                            int(current_best[1]-(ncols-1)/2):int(current_best[1]+(ncols-1)/2)+1]
        return fixed_patch, tuple(current_best[0:2]), int(current_best[2])

    def display_panel(self, moving="", critical_idx="", iternum="", score="", cmap='hot', title_list = ["","",""],savename='fit.png'):
        """
        Plot simulations inside the experimental image after fitting

        Args:
            moving: (numpy array) image that will be fit inside experiment
            critical_idx: (tuple) position where moving image will be set inside experiment
            iternum: (int) optional, denotes the iteration that the fit was found, to appear as text
            score: (int) optional, denotes the score of the fit, to appear as text
            cmap: (string) choice of color map for display
            title_list: (list) list of titles given to each panel
            savename: (string) saveas name for display image
        """

        plt.rcParams["font.family"] = "Arial"

        fixed  = self.exp_img.copy()
        fixed  = iop.scale_pixels(fixed, mode='grayscale')
        moving = iop.scale_pixels(moving, mode='grayscale')
        
        nrows, ncols = np.shape(moving)
        rcrds = iop.pixels_within_rectangle(int(critical_idx[1]-(ncols-1)/2), int(critical_idx[0]-(nrows-1)/2), ncols, nrows)

        base_img = fixed.copy()
        filler = moving.flatten("F")

        for i in range(len(rcrds)):
            entry = rcrds[i]
            base_img[entry[1],entry[0]] = filler[i]
        
        # Likely to make simulation square (not always good). Can delete this!
        # moving = iop.apply_resize(moving, np.shape(fixed))

        # Will make heights the same but keep aspect ratio
        fac = np.shape(fixed)[0]/np.shape(moving)[0]
        new_size = np.round((fac * np.array(np.shape(moving))),0).astype(np.int)
        moving = iop.apply_resize(moving, new_size)

        # Add border to simulation so appears uniform next to experiment (sizing)
        try:
            # Will work if fixed smaller than moving
            brdx = 255*np.ones((int(np.shape(moving)[0]),int(np.floor((np.shape(fixed)[1] - np.shape(moving)[1])/2))))
            moving = np.hstack([brdx,moving,brdx])
        except:
            pass
        
        # Ensure that all images are the same size (same as experiment)
        moving = iop.apply_resize(moving, np.shape(fixed))
        base_img = iop.apply_resize(base_img, np.shape(fixed))

        fig, axes = plt.subplots(nrows=1, ncols=3)
        fig.add_gridspec(nrows=1, ncols=3).update(wspace=0.0, hspace=0.0)
  
        axes[0].imshow(fixed, interpolation='quadric', cmap=cmap)
        axes[0].set_title(title_list[0],fontsize=8.5)
        axes[0].axis('off')

        axes[1].imshow(moving  , interpolation='quadric', cmap=cmap)
        axes[1].set_title(title_list[1],fontsize=9)
        axes[1].axis('off')

        axes[2].imshow(base_img, interpolation='quadric', cmap=cmap)
        axes[2].set_title(title_list[2],fontsize=9)
        axes[2].axis('off')
        
        pnt1 = np.min(rcrds,axis=0)
        pnt2 = np.max(rcrds,axis=0)
        axes[2].plot([pnt1[0],pnt1[0]],[pnt1[1],pnt2[1]],lw=0.5,color="#8AFF30")
        axes[2].plot([pnt2[0],pnt2[0]],[pnt1[1],pnt2[1]],lw=0.5,color="#8AFF30")
        axes[2].plot([pnt1[0],pnt2[0]],[pnt1[1],pnt1[1]],lw=0.5,color="#8AFF30")
        axes[2].plot([pnt1[0],pnt2[0]],[pnt2[1],pnt2[1]],lw=0.5,color="#8AFF30")

        if score != "":
            axes[2].text(0.53*np.shape(base_img)[1],1.07*np.shape(base_img)[0],"FOM: "+"{:8.5f}".format(float(score)),va='center',ha='left',fontsize=9)
        
        if iternum != "":
            axes[2].text(0,1.07*np.shape(base_img)[0],"iteration: "+str(iternum),va='center',ha='left',fontsize=9)

        plt.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.02)
        fig.set_size_inches((6.5, 2.4), forward=True)

        plt.savefig(savename,dpi=400)
        plt.close()

    def taxicab_ssim_objective(self,x):
        """
        Objective function used to quantify how well a given set of input imaging paramters, x, 
        produce an image that can be arranged in such a way inside the experimental image that minimizes 
        the custom taxicab_ssim objective function: 
        
            𝐽(𝜃)= 𝛼*d_TC(𝜃) + 𝛽*d_𝑆𝑆𝐼𝑀(𝜃)  
    
            - where d_TC(𝜃) is the taxicab distance required for optimal cross-correlation-based 
              registration after upsampling (enforces consistency in the pattern across boundaries)

            - where d_SSIM(𝜃) is the Structural Similarity Index Measure, which quantifies the visual 
              similarity between the simulation and experiment patch (enforces visual consistency in 
              the content within the patches)
              
        Args:
            x: (np.array) set of parameters to test

        Returns:
            The objective function value (refered to as the the "figure-of-merit" value).
        """
        xfit = [a for a in x[:-2]] + [int(a) for a in x[-2::]]
        try:
            sim_img, sim_struct, exp_patch, shift_score, __ = self.fit(xfit, display=False);
        except Exception as e: 
            print(e)
            sim_img = None
        
        if sim_img is not None:
            match_ssim = iop.score_ssim(sim_img, exp_patch, win_size=35)
            fom = 0.1*(shift_score)+match_ssim
        else:
            fom = 9999
        summary = self.sim_obj.simulation_summary(self.iter)
        summary = summary + "\n       🌀 FOM                       :  {}\n".format(fom)
        # Print for viewing progress
        print(summary)
        # Print to record progress to file
        print(','.join(str(v) for v in [self.iter] + xfit + [fom]), file=open("progress.txt", "a"))
        self.iter += 1
        return fom

    def taxicab_ssim_objective_gb(self,x):
        """
        Objective function used to quantify how well a given set of input imaging paramters, x, 
        produce an image that can be arranged in such a way inside the experimental image that minimizes 
        the custom taxicab_ssim objective function: 
        
            𝐽(𝜃)= 𝛼*d_TC(𝜃) + 𝛽*d_𝑆𝑆𝐼𝑀(𝜃)  
    
            - where d_TC(𝜃) is the taxicab distance required for optimal cross-correlation-based 
              registration after upsampling (enforces consistency in the pattern across boundaries)

            - where d_SSIM(𝜃) is the Structural Similarity Index Measure, which quantifies the visual 
              similarity between the simulation and experiment patch (enforces visual consistency in 
              the content within the patches)
              
        Uses special "fit_gb" which prioritizes seach closer to the center (horizontal) of the image.

        Args:
            x: (np.array) set of parameters to test

        Returns:
            The objective function value (refered to as the the "figure-of-merit" value).
        """
        xfit = [a for a in x[:-2]] + [int(a) for a in x[-2::]]
        try:
            sim_img, sim_struct, exp_patch, shift_score, __ = self.fit_gb(xfit, display=False);
        except Exception as e: 
            print(e)
            sim_img = None

        if sim_img is not None:
            match_ssim = iop.score_ssim(sim_img, exp_patch, win_size=35)
            fom = 0.1*(shift_score)+match_ssim
        else:
            fom = 9999
        summary = self.sim_obj.simulation_summary(self.iter)
        summary = summary + "\n       🌀 FOM                       :  {}\n".format(fom)
        # Print for viewing progress
        print(summary)
        # Print to record progress to file
        print(','.join(str(v) for v in [self.iter] + xfit + [fom]), file=open("progress.txt", "a"))
        self.iter += 1
        return fom

    def find_correspondence(self,objective='taxicab_ssim', optimizer='Powell', initial_solution="", search_mode="stm"):
        """
        Wrapper around scipy.optimize.minimize solvers, to find optimal correspondence between simulation and experiment 
        by minimizing taxicab_ssim_objective.
        
        Args:
            objective: (string) 
            optimizer: (string) 
            initial_solution: (np.array)
            constraint_list: (list)
            search_mode: (string) Specify search mode ('gb' for STEM grain boundaries, and 'stm' for STM images)
            
        Returns:
        """
        if os.path.isfile(os.getcwd()+"/progress.txt"):
            proceed = ""
            while str(proceed).upper() not in ["Y","N"]:
                proceed = input("Append to existing progress file? [Y/N]: ")
            if str(proceed).upper() == "N":
                os.remove(os.getcwd()+"/progress.txt")
            else:
                pass
        
        self.iter = 1
        # Keep this in because its useful for constraints that need to be enforced by the optimizer.
        # Currently, constraints are enforced by a combination of clamping (in sim_params), or throwing exceptions in the 
        # image simulation steps
        constraints = []
        # def make_constraint_list(c):
        #    constraints = []
        #    for idx in range(len(c)):
        #        constraints.append({'type': 'ineq', 'fun': lambda t, idx=idx: t[idx] - c[idx][0]})
        #        constraints.append({'type': 'ineq', 'fun': lambda t, idx=idx: c[idx][1] - t[idx]})
        #    return constraints
        print("Search mode: {}".format(search_mode))
        if search_mode == 'stm':
            if optimizer == 'COBYLA':
                # constraints = make_constraint_list(constraint_list)
                if objective  == "taxicab_ssim":
                    return minimize(self.taxicab_ssim_objective, initial_solution, method='COBYLA',tol=1E-6,options={'disp': True, 'rhobeg': 0.25, 'catol': 0.01}, constraints=constraints)
            
            if optimizer == 'Powell':
                if objective  == "taxicab_ssim":
                    return minimize(self.taxicab_ssim_objective, initial_solution, method='Powell',tol=1E-6,options={'disp': True})

        elif search_mode == 'gb':
            if optimizer == 'COBYLA':
                # constraints = make_constraint_list(constraint_list)
                if objective  == "taxicab_ssim":
                    return minimize(self.taxicab_ssim_objective_gb, initial_solution, method='COBYLA',tol=1E-6,options={'disp': True, 'rhobeg': 0.25, 'catol': 0.01}, constraints=constraints)
            
            if optimizer == 'Powell':
                if objective  == "taxicab_ssim":
                    return minimize(self.taxicab_ssim_objective_gb, initial_solution, method='Powell',tol=1E-6,options={'disp': True})

        else:
            print("Search mode {} not understood! Select either 'gb' for STEM grain boundaries, or 'stm' for STM images")