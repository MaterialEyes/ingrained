import time
import sys, os

sys.path.append("../../../")
import numpy as np
import matplotlib.pyplot as plt
import ingrained.image_ops as iop
from ingrained.structure import Bicrystal, PartialCharge
from ingrained.optimize import CongruityBuilder
from multiprocessing import Pool

class Printer:
    """Print things to stdout on one line dynamically"""

    def __init__(self, data):
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
    
    
def compareAngles(z_below,z_above,sim_obj,exp_img,interval=10):
    """
    Function to quickly test which angle is the best fit for the system.
    As no optimiziation occurs, one should be careful when using this
    function to determine the best initial angle for the full ingrained
    optimization run.
    
    Inputs:
        z_below (float): Thickness or depth from top in (Angstrom)
        z_above (float): Distance above the surface to consider (Angstrom)
        sim_obg (ingrained.structure): PartialCharge object
        exp_img (ingrained.image_ops): Experimental image fit against
        interval (int): The interval of angles to rotate through
    """
    
    rhos, nzmax = sim_obj._get_stm_vol(z_below,z_above)
    
    score_list=[]
    for ang in range(0,360,interval):
        cb = CongruityBuilder(sim_obj=sim_obj,exp_img=exp_img['Pixels'])
        score = cb.taxicab_ssim_objective([z_below,z_above,rhos.max()*3/4,
                                              rhos.max()*3/4*.99,
                                              0,0,0,0,
                                              ang,
                                              exp_img["Experiment Pixel Size"],
                                              0,
                                              int(len(exp_img['Pixels'])/2),
                                              int(len(exp_img['Pixels'][0])/2)
                                              ],
                                              fixed=[[[]]],
                                              append_summary=False)
        score_list.append([score,ang])
    return(score_list)
    
    

def multistart(sim_params,num_starts,sim_obj,exp_img,
                objective='taxicab_ssim',optimizer='Powell',search_mode='',
                fixed_params=[]):
    """
    Function to facilitate running multiple optimizations at once
    NOTE: Will not function properly if progress files already exist!!!



    Args:
        sim_params (list): The initial parameters to use
        num_starts (int) : The number of optimizations to run
                           in parallel
        sim_obj (PartialCharge): PARCHG to use as reference
        exp_img (2x2 array)    : Array representing the
                                 experimental image
        fixed_params (list): List of sim_param indexes to keep fixed

    """
    starts=[]
    rand = np.random
    if search_mode=='stm':
        for i in range(num_starts):
            zcap = .25*sim_obj.structure.lattice.c
            new_input=[sim_params[0]*rand.random(),
                       zcap/2+zcap/2*rand.random()]
            rhos, nzmax = sim_obj._get_stm_vol(new_input[0],
                                               new_input[1])
            new_input.append(rhos.max()/3+rhos.max()*2/3*rand.random())
            new_input.append(new_input[2]*.98)
            new_input+=sim_params[4:10]
            new_input.append(1+2*rand.random())
            new_input+=[sim_params[11]*.9+sim_params[11]*.2*rand.random(),
                        sim_params[12]*.9+sim_params[12]*.2*rand.random()]
            new_input+=[i,sim_obj, exp_img,
                        fixed_params,
                        objective,optimizer,search_mode]
            starts.append(new_input)
    elif search_mode=='gb':
        for i in range(num_starts):
            new_input=[sim_params[0]*.95+sim_params[0]*.1*rand.random(),
                       rand.random(),
                       2*rand.random()]
            new_input+=sim_params[4:8]
            new_input+=[sim_params[9]*.9+sim_params[9]*.2*rand.random(),
                        sim_params[10]*.9+sim_params[10]*.2*rand.random()]
            new_input+=[i,sim_obj, exp_img,
                        fixed_params,
                        objective,optimizer,search_mode]

    print(len(starts[0]))
    workers = Pool(processes=num_starts)
    workers.map(multi_congruity_finder, starts)

def multistart_series(sim_params,num_starts,sim_obj,exp_img,
                objective='taxicab_ssim',optimizer='Powell',search_mode='',
                fixed_params=[],path=''):
    """
    Function to facilitate running multiple optimizations at once
    NOTE: Will not function properly if progress files already exist!!!



    Args:
        sim_params (list): The initial parameters to use
        num_starts (int) : The number of optimizations to run
                           in parallel
        sim_obj (PartialCharge): PARCHG to use as reference
        exp_img (2x2 array)    : Array representing the
                                 experimental image
        fixed_params (list): List of sim_param indexes to keep fixed

    """
    starts=[]
    rand = np.random
    if search_mode=='stm':
        for i in range(num_starts):
            zcap = .25*sim_obj.structure.lattice.c
            new_input=[sim_params[0]*rand.random(),
                       zcap/2+zcap/2*rand.random()]
            rhos, nzmax = sim_obj._get_stm_vol(new_input[0],
                                               new_input[1])
            new_input.append(rhos.max()/3+rhos.max()*2/3*rand.random())
            new_input.append(new_input[2]*.98)
            new_input+=sim_params[4:10]
            new_input.append(1+2*rand.random())
            new_input+=[sim_params[11]*.9+sim_params[11]*.2*rand.random(),
                        sim_params[12]*.9+sim_params[12]*.2*rand.random()]
            new_input+=[i,sim_obj, exp_img,
                        fixed_params,
                        objective,optimizer,search_mode]
            starts.append(new_input)
    elif search_mode=='gb':
        for i in range(num_starts):
            new_input=[sim_params[0]*.95+sim_params[0]*.1*rand.random(),
                       rand.random(),
                       2*rand.random()]
            new_input+=sim_params[4:8]
            new_input+=[sim_params[9]*.9+sim_params[9]*.2*rand.random(),
                        sim_params[10]*.9+sim_params[10]*.2*rand.random()]
            new_input+=[i,sim_obj, exp_img,
                        fixed_params,
                        objective,optimizer,search_mode]

    multi_congruity_finder_series(starts,path)



def multistart_stm_angle(sim_params,num_starts,sim_obj,exp_img,
                objective='taxicab_ssim',optimizer='Powell',
                fixed_params=[],interval=30,cap=180):
    """
    Function to facilitate running multiple optimizations at once,
    focused on rotating the initial image with the same paramenters
    NOTE: Will not function properly if progress files already exist!!!



    Args:
        sim_params (list): The initial parameters to use
        num_starts (int) : The number of optimizations to run
                           in parallel
        sim_obj (PartialCharge): PARCHG to use as reference
        exp_img (2x2 array)    : Array representing the
                                 experimental image
        fixed_params (list): List of sim_param indexes to keep fixed

    """
    starts=[]
    for i in range(0,cap+1,interval):
            new_input=list(sim_params)
            new_input[9]=i
            new_input+=[i,sim_obj, exp_img,
                        fixed_params,
                        objective,optimizer,'stm']
            starts.append(new_input)


    workers = Pool(processes=num_starts)
    workers.map(multi_congruity_finder, starts)


def multi_congruity_finder(start_inputs):
    """
    Helper for the 'multistart' function to allow parallelization
    """
    if start_inputs[-1]=='stm':
        initial_solution=start_inputs[:13]
        (counter,sim_obj,exp_img,fixed_index,
             objective,optimizer,search_mode)=start_inputs[13:]
    elif start_inputs[-1]=='gb':
        initial_solution=start_inputs[:11]
        (counter,sim_obj,exp_img,fixed_index,
             objective,optimizer,search_mode)=start_inputs[11:]
        
    congruity = CongruityBuilder(sim_obj=sim_obj, exp_img=exp_img)
    congruity.find_correspondence(objective=objective, optimizer=optimizer,
                                  initial_solution=initial_solution,
                                  fixed_params=fixed_index,
                                  search_mode=search_mode,
                                  counter=counter)

def multi_congruity_finder_series(start_inputs_list,path):
    """
    Helper function for the 'multistart_series' function to 
    run ingrained in series
    
    Variables:
        path (str): Path to where to write the files
    """
    for start_inputs in start_inputs_list:
        print(start_inputs)
        if start_inputs[-1]=='stm':
            initial_solution=start_inputs[:13]
            (counter,sim_obj,exp_img,fixed_index,
                 objective,optimizer,search_mode)=start_inputs[13:]
        elif start_inputs[-1]=='gb':
            initial_solution=start_inputs[:11]
            (counter,sim_obj,exp_img,fixed_index,
                 objective,optimizer,search_mode)=start_inputs[11:]
        congruity = CongruityBuilder(sim_obj=sim_obj, exp_img=exp_img)
        congruity.find_correspondence(objective=objective, optimizer=optimizer,
                                      initial_solution=initial_solution,
                                      fixed_params=fixed_index,
                                      search_mode=search_mode,
                                      counter=counter,path=path)





def locate_frame(idx,progress,search_mode,congruity,
                         cmap,describe_frames,save_path,bias_y=''):
    """
    Helper function for repetition in 'print_frames'
    Args:
        idx (int): index
    """
    idx = int(idx)
    # Select data from frame
    x = progress[idx]
    xfit = x[1:-1]
    xfit = [a for a in xfit[:-2]] + [int(a) for a in xfit[-2::]]
    blockPrint()
    try:
        if search_mode.lower() == "gb":
            (sim_img, sim_struct, exp_patch, shift_score, stable_idxs,
            ) = congruity.fit_gb(sim_params=xfit, bias_y=bias_y)
        elif search_mode.lower() == "stm":
            (sim_img, sim_struct, exp_patch, shift_score, stable_idxs,
            ) = congruity.fit(sim_params=xfit)
    except:
        sim_img = np.ones(np.shape(congruity.exp_img))
        stable_idxs = (0, 0)
    congruity.display_panel(
        moving=sim_img,
        critical_idx=stable_idxs,
        score=x[-1],
        iternum=int(x[0]),
        cmap=cmap,
        title_list=describe_frames,
        savename=save_path
        + "/frames/ingrained"
        + "-"
        + str(idx).zfill(5)
        + ".png",
    )


def height_profiles(sim_img,pix_size,prec=1E-3,center_index=0,resolution=40):
        """
        Obtain information on the height profiles between peaks        

        Args:
            resolution (int): The number of points to sample in the profile
            pix_size (float): The width of image over number of pixels
            save (Boolean): Whether to save the height profile as an image

        Returns:
            list (list): List of heights, terminated by length of the profile
        """
        
        img=sim_img
        max_ind=[]
        # Get the maximum height in the array
        max_height = np.matrix(img).max()
        for i in range(len(img)):
            for j in range(len(img[i])):
                # If the height of the index is within a tolerance
                if max_height-img[i][j]<prec:
                    max_ind.append(np.array([i,j]))
        all_profiles = []
        # Using the first index as the starting point
        for ind in [i for i in range(len(max_ind)) if i!=center_index]:
            profiles = []
            start = max_ind[ind]
            end   = max_ind[center_index]
            start,end = np.array(start),np.array(end)
            for i in range(resolution+1):

                new_ind = (end-start)*i/40+start
                # Get the index bounds
                top, bot, left, right = int(new_ind[0]),\
                                        int(new_ind[0]-1),\
                                        int(new_ind[1]-1),\
                                        int(new_ind[1])
                # Scale using lever rule
                val=sum([img[top][left]*(new_ind[0]-bot)*(right-new_ind[1]),
                         img[top][right]*(new_ind[0]-bot)*(new_ind[1]-left),
                         img[bot][left]*(top-new_ind[0])*(right-new_ind[1]),
                         img[bot][right]*(top-new_ind[0])*(new_ind[1]-left)])
                profiles.append(val)
            length = np.linalg.norm(end-start)*pix_size
            profiles.append(length)
            all_profiles.append(profiles)
        return(all_profiles)
    
    

def print_frames(config_file="", poscar_file="", exp_img="", exp_title="",
                     progress_file="", frame_selection="", 
                     search_mode="", cmap="hot"):

    if not isinstance(frame_selection, str):
        print(
            "Frame selection must be provided as a string! \
            (i.e. 'all' or '3,8,26', or '1-10', or '1:3:300', etc.)"
        )
        raise ValueError("Invalid user response")

    if frame_selection == "":
        # TODO: include error checking on input
        decision = input(
            """Select frames to write to a display panel image

        Example options
            all    : all frames
            best   : best frame
            final  : final frame
            10     : frame #10
            3,8,26 : multiple individual frames (#3, #8, #26)
            4-74   : frames within a range (#4 through and including #74)
            1:2:30 : start, step, stop sequence of frames
            
        >> Selection:  """
        )

    else:
        decision = frame_selection

    if search_mode.lower() == "gb":
        # Initialize a Bicrystal object with the path to the slab json file
        if poscar_file != "":
            bicrystal = Bicrystal(poscar_file="bicrystal.POSCAR.vasp")
        else:
            bicrystal = Bicrystal(config_file=config_file)

        describe_frames = [exp_title, "HAADF simulation", "$ingrained$ fit"]
        sim_obj = bicrystal
        bias_y = 0.15

    elif search_mode.lower() == "stm":
        # Initialize a PartialCharge object with the path to the PARCHG file
        parchg = PartialCharge(config_file=config_file)
        parchg._shift_sites()
        parchg._shift_sites()
        describe_frames = [exp_title, "STM simulation", "$ingrained$ fit"]
        sim_obj = parchg
        bias_y = 0.00

    else:
        print(
            "Search mode {} not understood! Select either 'gb' for \
            STEM grain boundaries, or 'stm' for STM images"
        )
        raise ValueError("Invalid user response")

    # Initialize a ConguityBuilder with PARCHG and experimental image
    congruity = CongruityBuilder(sim_obj=sim_obj, exp_img=exp_img)

    # Get solutions from text file
    progress = np.genfromtxt(progress_file, delimiter=",")

    save_path = os.path.dirname(os.path.realpath(progress_file))
    os.makedirs(save_path + "/frames/", exist_ok=True)

    if decision.lower() == "all":
        for i in range(np.shape(progress)[0]):
            locate_frame(i,progress,search_mode,
                              congruity,cmap,describe_frames,save_path,bias_y)

    elif decision.lower() == "best":
        best_idx = int(np.argmin(progress[:, -1]))
        locate_frame(best_idx,progress,search_mode,
                              congruity,cmap,describe_frames,save_path,bias_y)

    elif decision.lower() == "final":
        final_idx = np.shape(progress)[0] - 1
        locate_frame(final_idx,progress,search_mode,
                              congruity,cmap,describe_frames,save_path,bias_y)
     
    elif (
        len(str(decision.lower()).split(",")) > 1
        and len(str(decision.lower()).split("-")) == 1
    ):
        for idx in str(decision.lower()).split(","):
            locate_frame(idx,progress,search_mode,
                              congruity,cmap,describe_frames,save_path,bias_y)
           
    elif (
        len(str(decision.lower()).split("-")) == 2
        and len(str(decision.lower()).split(",")) == 1
    ):
        lb, ub = str(decision.lower()).split("-")
        for idx in range(int(lb), int(ub) + 1):
            locate_frame(idx,progress,search_mode,
                              congruity,cmap,describe_frames,save_path,bias_y)
     
    elif (
        len(str(decision.lower()).split(":")) == 3
        and len(str(decision.lower()).split("-")) == 1
        and len(str(decision.lower()).split(",")) == 1
    ):
        start, step, stop = str(decision.lower()).split(":")
        for idx in range(int(start), int(stop) + 1, int(step)):
            locate_frame(idx,progress,search_mode,
                              congruity,cmap,describe_frames,save_path,bias_y)
           
    elif decision.isdigit():
        locate_frame(idx,progress,search_mode,
                              congruity,cmap,describe_frames,save_path,bias_y)
           
    else:
        Printer('Selection "{}" not understood!'.format(decision))
        print()

    time.sleep(1)
    Printer("")



def prepare_fantastx_input(
    config_file="", poscar_file="", exp_img="", progress_file=""
):

    if poscar_file != "":
        # Initialize a Bicrystal object with the path to the initial poscar
        sim_obj = Bicrystal(poscar_file=poscar_file)
    else:
        # Initialize a Bicrystal object with the path to the slab json file
        sim_obj = Bicrystal(config_file=config_file)

    bias_y = 0.15

    # Initialize a ConguityBuilder with PARCHG and experimental image
    congruity = CongruityBuilder(sim_obj=sim_obj, exp_img=exp_img)

    # Get solutions from text file
    progress = np.genfromtxt(progress_file, delimiter=",")

    save_path = os.path.dirname(os.path.realpath(progress_file))
    os.makedirs(save_path + "/fantastx_start/", exist_ok=True)

    best_idx = int(np.argmin(progress[:, -1]))
    x = progress[best_idx]
    xfit = x[1:-1]
    xfit = [a for a in xfit[:-2]] + [int(a) for a in xfit[-2::]]

    try:
        sim_img, sim_struct, exp_patch, shift_score, stable_idxs = \
                               congruity.fit_gb(sim_params=xfit, bias_y=bias_y)
    except:
        sim_img = np.ones(np.shape(congruity.exp_img))

    sim_struct.to(filename=save_path + "/fantastx_start/" + "ingrained.POSCAR.vasp")

    match_ssim = iop.score_ssim(sim_img, exp_patch, win_size=35)
    print("FOM: {}".format(match_ssim))

    sim_params = xfit.copy()
    sim_params[1] = 0
    np.savetxt(save_path + "/fantastx_start/" + "sim_params", sim_params)

    np.save(save_path + "/fantastx_start/" + "experiment.npy", exp_patch)
