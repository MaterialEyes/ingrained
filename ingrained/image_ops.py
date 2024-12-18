import os
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from skimage.draw import polygon
from skimage.filters import gaussian
from skimage.metrics import structural_similarity as ssim

# for pySPM, might need to run:
# >> pip install pySPM
import pySPM

# for dm3_lib, might need to run:
# >> git clone https://bitbucket.org/piraynal/pydm3reader/;
# >> cd pydm3reader/;
# >> pip install .;
# >> cd ..;
# >> rm -rf pydm3reader;
import dm3_lib as dm3lib


def getTipH(w,h,r):
    """
    Helper function for getting height difference between
    the tip at distance x from the tip center and the
    raised feature of height h from the proposed tip end 
    point

    Args:

        w (float): lateral distance between tip center and the peak
           be sure to subtract a half-pixel width, Aangstrom
        
        h (float): height difference between proposed tip end point 
           and the peak height, Aangstrom

        r (float): radius of the tip, Aangstrom

    Returns:
        
        (float): The difference in position between the 
                 curve of the tip and the peak height;
                 negative values indicate the tip and 
                 peak are overlapping

    """

    if r<=w:
        y_tip=h
    else:
        #y_tip = np.sin(np.arccos(w/r))*r
        y_tip = r-np.cos(np.arcsin(w/r))*r

    return(y_tip-h)

def sq_root(x1,x2,y1,y2):
    """
    helper funciton for getting square root
    """
    return(((x2-x1)**2+(y2-y1)**2)**.5)

def apply_tip_smoothing(img,radius,width,prec=1,bounds=[0,-1]):
    """
    Apply smoothing to STM surface plot as if a physical
    STM tip of smoothness radius were used to measure it

    Args:    

        img (NxM array): the image to be smoothed

        radius (float): the radius of the tip, in Aangstroms

        width (float): how many pixels wide is an Aangstrom; is also 1/pix_size

        prec (float): What is the maximum height above the lowest point
                      to consider for re-evaluation
                      
        bounds (list, 1x2): The range of pixels to consider
                            when re-evaluating heights. Useful
                            for when rate limiting peaks are 
                            over the edge of the original image

    Returns:
        
        img (NxM array): the re-scaled image

    """


    array = img
    min_indy,min_indx=np.unravel_index(img.argmin(), img.shape)    

    min_ind=[]
    min_height = np.matrix(img).min()


    # Get the tip radius in pixel units
    pixRad = max(1,round(radius*width))

    if bounds[0]<0:
        bounds[0]=len(array)+bounds[0]+1
    if bounds[1]<0:
        bounds[1]=len(array)+bounds[1]+1
    # Get list of indexes to check the new height of
    for i in range(bounds[0],bounds[1]):
        for j in range(bounds[0],bounds[1]):
            # If the height of the index is within a tolerance
            # in units of Aangstrom
            if img[i][j]-min_height<=prec:
                min_ind.append(np.array([i,j]))

    newArray=np.array(array)
    arrLen=max(len(array),len(array[0]))
    # If the diameter of the tip is less than the
    # width of a pixel, no smoothing can occur
    if 2*radius<1/width:
        return(array)
    for site in min_ind:
        sy,sx=site
        subArray=[]
        # Get list of heights for x-y positions within the 
        # range of the tip, along with coordinates
        if radius>=sq_root(0,arrLen/width,0,arrLen/width):
            for j in range(0,arrLen):
                for i in range(0,arrLen):
                    subArray.append([array[j][i],[j,i]])
        else:
            for j in range(max(0,sy-pixRad),min(arrLen,sy+pixRad)):
                for i in range(max(0,sx-pixRad),min(arrLen,sx+pixRad)):
                    if sq_root(j,sy,i,sx)/width<=radius:
                        subArray.append([array[j][i],[j,i]])
        propTipH=array[sy][sx]
        subArray.sort()


        # Loop over peaks to find at what tip height
        # above (sy,sx) that makes contact with the
        # surface of the material
    
        while True:
            newHs=[]
            for h,index in subArray:
                iy,ix=index
                distFromTip=sq_root(ix,sx,iy,sy)/width
                tempH=getTipH(distFromTip,
                                h-propTipH,
                                radius)
                newHs.append([round(tempH,6),h,index])
            newHs.sort()
            # If tip is touching the rate limiting
            # peak, end the loop
            if -1E-4<newHs[0][0]<1E-4:
                break
            # Else, shift tip to be closer to peak
            else:
                propTipH-=newHs[0][0]
        

        ref1=newHs[0]

        # Find the new center of the tip
        sphereCenter=(radius**2-((ref1[2][0]-sy)/width)**2-\
                        ((ref1[2][1]-sx)/width)**2)**.5+ref1[1]
        # Pixel height is either the new position of the tip,
        # or the original height
        newArray[sy][sx]=max(sphereCenter-radius,array[sy][sx])
            

    # Set the lowest position pixel to be constant
    # to ensure proper contrast
    newArray[min_indy][min_indx]=min_height
    return(newArray)



def apply_rotation(img, rotation):
    """
    Apply centered rotation to an image and crop collars.

    Args:
        img: (np.array)
        rotation: (float) rotation angle in degrees in counter-clockwise direction

    Returns:
        A numpy array (np.float64) of the rotated image.
    """
    # Rotate tile according to rotation value
    img_rotation = transform.rotate(img, rotation, preserve_range=True)

    # Find edge_length for fully inscribed square
    edge_length = (
        int(
            np.min(np.shape(img)) / (
                np.cos(np.radians(np.abs(rotation) % 90))
                + np.sin(np.radians(np.abs(rotation) % 90)))
            ) - 5
    )

    # Return square image with collars from rotation removed
    return apply_crop(img_rotation, edge_length, edge_length)


def apply_shear(img, x_sh, y_sh):
    """
    Apply centered shear to an image and crop collars.

    TODO: The collars are cropped conservatively (more than necessary...)
    Implement equation for largest inscribed square to return largest image.

    Args:
        img: (np.array)
        x_sh: (float) fractional amt shear in x
        y_sh: (float) fractional amt shear in y

    Returns:
        A numpy array (np.float64) of the sheared image.
    """
    # Get size of image for centering translation in transform
    nrows, ncols = np.shape(img)

    # 2D affine transformation
    afine_tf = transform.AffineTransform(
        matrix=np.array(
            [[1, x_sh, -(ncols * x_sh) / 2], 
             [y_sh, 1, -(nrows * y_sh) / 2], 
             [0, 0, 1]]
        )
    )

    # Apply transform to image data
    img_shear = transform.warp(img, inverse_map=afine_tf)

    # Find edge_length for a fully inscribed square 
    # (this is a conservative approach)
    edge_length = int(
        np.min(
            [
                (np.shape(img)[1] - np.shape(img)[1] * np.abs(x_sh)),
                (np.shape(img)[0] - np.shape(img)[0] * np.abs(y_sh)),
            ]
        )
    )

    # Return square image with collars from shear removed
    return apply_crop(img_shear, edge_length, edge_length)


def apply_stretch(img, x_st, y_st):
    """
    Apply stretch or squeeze to an image and crop based on smallest dimension.

    Args:
        img: (np.array)
        x_st: (float) fractional amt stretch (+) or compression (-) in x
        y_st: (float) fractional amt stretch (+) or compression (-) in y

    Returns:
        A numpy array (np.float64) of the stretched image.
    """
    # Get current size of image
    nrows, ncols = np.shape(img)

    # Apply resize so as to stretch/squeeze image
    img_stretch = transform.resize(
        img,
        (int((1 + y_st) * nrows), int((1 + x_st) * ncols)),
        preserve_range=True,
        anti_aliasing=True,
    )

    # Find edge_length for inscribed square
    edge_length = np.min(np.shape(img_stretch))

    # Return square image
    return apply_crop(img_stretch, edge_length, edge_length)


def apply_crop(img, x_cr, y_cr):
    """
    Apply centered crop to an image.

    Args:
        img: (np.array)
        x_cr: (int) final x dimension (crop to)
        y_cr: (int) final y dimension (crop to)

    Returns:
        A numpy array (np.float64) of the cropped image.
    """
    y, x = img.shape
    startx = x // 2 - x_cr // 2
    starty = y // 2 - y_cr // 2
    return img[starty : starty + y_cr, startx : startx + x_cr]


def apply_resize(img, rs):
    """
    Apply range preserving resize to an image (with anti-aliasing).

    Args:
        img: (np.array)
        rs: (tuple) new size (row,col)

    Returns:
        A numpy array (np.float64) of the resized image.
    """
    return transform.resize(img, rs, preserve_range=True, anti_aliasing=True)


def apply_blur(img, sigma=0):
    """
    Apply gaussian blur to an image.

    Args:
        img: (np.array)
        sigma: (float) standard deviation for gaussian kernel

    Returns:
        A numpy array (np.float64) of the blurred image.
    """
    return gaussian(img, sigma=sigma)


def apply_quantize_downsample(img, factor=2):
    """
    Quantize pixel values (16 levels of greyscale) and downsample image by
    factor (reduce spatial resolution)

    Args:
        img: (numpy array)
        factor: (int) keeping only every nth sample

    Returns:
        A square numpy array (np.uint8) of the downsampled and quantized image
    """
    # Cast pixel values to float64 to ensure precision
    img = img.astype(np.float64)

    # Rescale pixels values to [0,1]
    img = (img - img.min()) / (img.max() - img.min())

    # Force pixels to assume one of 16 possible discrete values (quantize)
    img = ((255 * img) // 16).astype(np.uint8)

    # Ensure that image size is odd (for existence of true center pixel)
    new_size = np.array(
        [int(2 * np.floor(a / 2)+1) for a in (np.array(np.shape(img))/factor)]
    )
    return transform.resize(img,new_size,preserve_range=True).astype(np.uint8)


def pixels_within_rectangle(r0, c0, width, height):
    """
    Generate coordinates of pixels within rectangle.

    Args:
        r0: uppermost row
        c0: leftmost column

    Returns:
         A list of tuple coordinates for the rectangle
    """
    rr, cc = [r0, r0 + width, r0+width, r0], [c0, c0, c0 + height, c0+height]
    pg = polygon(rr, cc)
    return list(zip(pg[0], pg[1]))


def scale_pixels(img, mode=None):
    """
    Select a pixel scaling technique

    Args:
        mode: (string) pixel scaling technique
          rescale    :  stretch the pixel intensities so to fill the 
                        range from 0 to 1 (float64)
          center     :  enforce zero mean, unit variance (float64)
          grayscale  :  stretch the pixel intensities so to fill the 
                        range from 0 to 255 (uint8)

    Returns:
        A numpy array (either float64 or uint8) of the scaled image.
    """
    img = img.astype(np.float64)
    if mode == None:
        return img
    elif mode == "rescale":
        return ((img - img.min()) / (img.max() - img.min()) + 1e-16).astype(np.float64)
    elif mode == "center":
        return ((img - img.mean()) / (img.std())).astype(np.float64)
    elif mode == "grayscale":
        return (255 * (img - img.min()) / (img.max() - img.min()) + 1e-16).astype(
            np.uint8
        )


def score_ssim_deprecated(img1, img2, win_size=35):
    """
    Compute the mean structural similarity index between two images

    Args:
        img1, img2:  (ndarray) images
        win_size: (int) length of sliding window used in 
                        comparison (must be odd value)

    Return:
        1 - the mean structural similarity index (i.e. ΔSSIM)
    """
    img1 = scale_pixels(img1, mode="grayscale")
    img2 = scale_pixels(img2, mode="grayscale")
    return 1 - ssim(img1, img2, win_size=win_size)


def score_ssim(img1, img2):
    """
    Compute the mean structural similarity index between two images

    Args:
        img1, img2:  (ndarray) images

    Return:
        1 - the mean structural similarity index (i.e. ΔSSIM)
    """
    img1 = scale_pixels(img1, mode="rescale")
    img2 = scale_pixels(img2, mode="rescale")
    im_max, im_min = max(img1.max(), img2.max()), min(img1.min(), img2.min())
    return 1 - ssim(img1, img2, data_range=im_max-im_min)

def open_construction_file(slab_path):
    """
    Read input slab files (json) which provide bicrystal assembly parameters

    Args:
        slab_path (string) path to json file

    Return:
        A dict containing bicrystal assembly parameters
    """
    with open(slab_path, "r") as f:
        return json.loads(f.read())


def load_sim_params(param_file):
    """
    Load simulation parameters to a numpy array with proper entry types

    Args:
        param_file (string) path to param file

    Return:
        A numpy array of entries from param_file
    """
    xfit = np.loadtxt(param_file)
    return [a for a in xfit[:-2]] + [int(a) for a in xfit[-2::]]


def read_SXM_file(sxm_path):
    """
    Read a sxm file and store as numpy array within dictionary

    Args:
        sxm_path (string) path to file

    Return:
        A dict containing the raw pixel values and the pixel size information
    """
    to_angstrom = {"m": 1e10, "um": 10000, "nm": 10}
    if sxm_path.split(".")[-1] == "txt":
        with open(sxm_path, "r") as f:
            contents = f.read().split("\n")
            image_details = [
                a.strip("\n").split("#")[-1].strip() for a in contents[0:4]
            ]
            image_lines = [
                [float(b) for b in a.split("\t") if b != ""] for a in contents[4:-1]
            ]
            image = np.array(image_lines)
            d = {
                a.split(":")[0].strip(): a.split(":")[1].strip() for a in image_details
            }
            if d["Width"].split(" ")[1] == "nm":
                ps_width = (float(d["Width"].split(" ")[0]) * 10) / np.shape(image)[1]
            if d["Height"].split(" ")[1] == "nm":
                ps_height = (float(d["Height"].split(" ")[0]) * 10) / np.shape(image)[0]
    elif sxm_path.split(".")[-1] == "sxm":
        d = {}
        spm_obj = pySPM.SXM(sxm_path)
        image = spm_obj.get_channel("Z", direction="both", corr="slope").pixels
        spm_unit = spm_obj.size["real"]["unit"]
        ps_width = (
            spm_obj.size["real"]["y"] * to_angstrom[spm_unit] / np.shape(image)[1]
        )
        ps_height = (
            spm_obj.size["real"]["x"] * to_angstrom[spm_unit] / np.shape(image)[0]
        )
    else:
        print("Unrecognized file format {0}".format(sxm_path.split(".")[-1]))
    try:
        assert ps_width == ps_height
        # Get min pixel value of image and used to fill in any
        # pixels that are corrupt
        X_min = np.min(np.nan_to_num(image, 1e10))
        d["Experiment Pixel Size"] = ps_width
        d["Pixels"] = np.nan_to_num(image, nan=X_min)
        return d
    except:
        print("Image not loaded!")
        return None


def read_tif_file(tif_path):
    """
    Read a tif file and store as numpy array within dictionary

    Args:
        tif_path (string) path to file

    Return:
        A dict containing the raw pixel values and the pixel size information
    """
    d = {}
    d["Pixels"] = plt.imread(tif_path)[:, :]
    d["Experiment Pixel Size"] = ""
    return d


def read_png_file(png_path):
    """
    Read a png file and store as numpy array within dictionary

    Args:
        png_path (string) path to file

    Return:
        A dict containing the raw pixel values and the pixel size information
    """
    d = {}
    d["Pixels"] = plt.imread(png_path)
    d["Experiment Pixel Size"] = ""
    return d


def read_dm3_file(dm3_path):
    """
    Read a dm3 file and store as numpy array within dictionary

    Args:
        dm3_path (string) path to file

    Return:
        A dict containing the raw pixel values and the pixel size information
    """
    to_angstrom = {"m": 1e10, "um": 10000, "nm": 10}
    d = {}
    exp_unit = dm3lib.DM3(dm3_path).pxsize[1].decode("UTF-8")
    d["Pixels"] = dm3lib.DM3(dm3_path).imagedata
    d["Experiment Pixel Size"] = dm3lib.DM3(dm3_path).pxsize[0] * \
                                                          to_angstrom[exp_unit]
    return d


def image_open(filename):
    """
    Extract raw pixel values from common microscopy/stm imaging files
    (.dm3, .sxm, .txt) based on file extension

    Args:
        filename (string) path to file

    Return:
        A dict containing the raw pixel values and the pixel size information
    """
    if filename.split(".")[-1] in ["sxm", "txt"]:
        d = read_SXM_file(filename)
    elif filename.split(".")[-1] in ["dm3", "dm4"]:
        d = read_dm3_file(filename)  # Note: dm4 untested!
    elif filename.split(".")[-1] in ["tif"]:
        d = read_tif_file(filename)
    elif filename.split(".")[-1] in ["png", "jpg"]:
        d = read_png_file(filename)
    else:
        print(
            'Cannot read image from file. The extension "{}" is currently \
            not supported!'.format(filename.split(".")[-1])
        )
        d = None
    return d
