import cv2
import numpy as np
from skimage.draw import polygon
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from .external_functions import vifp_mscale

def pixel_value_rescale(img,dtype="uint8"):
    img = img.astype(np.float64)
    scaled = (img - img.min()) / (img.max() - img.min())
    if dtype=="uint8":
        img = (255*scaled).astype(np.uint8)
    elif dtype=="uint4": #4-bit
        img = ((255*scaled) // 16).astype(np.uint8)
    elif dtype=="float64": #4-bit
        img = ((255*scaled)).astype(np.float64)
    else:
        img = (scaled).astype(np.float32)
    return img

def crop_by_ratio(img,rrow,rcol):
    """Crop an image, specifying the border to remove as a ratio.
    Args:
        img: image array 
        rrow : tuple containing ratio of start/end row
        rcol : tuple containing ratio of start/end col
    Returns:
        A list of tuple coordinates for the rectangle
    """
    rrow_start, rrow_end = rrow
    rcol_start, rcol_end = rcol
    return img[int(np.floor(np.shape(img)[0]*rrow_start)):int(np.floor(np.shape(img)[0]*rrow_end)),\
               int(np.floor(np.shape(img)[1]*rcol_start)):int(np.floor(np.shape(img)[1]*rcol_end))]

def get_rectangle_crds(r0, c0, width, height):
    """Generate coordinates of pixels within rectangle.
    Args:
        r0: uppermost row 
        c0: leftmost column 
    Returns:
        A list of tuple coordinates for the rectangle
    """
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height, c0 + height]
    pg = polygon(rr, cc)
    return list(zip(pg[0],pg[1]))

def cutout_around_pixel(template,target,indx):
    """Generate coordinates of pixels within rectangle
    Args:
        template: the shape determining the size of the cutout
        target: the image the cutout is made from
        indx: the pixel to center the cutout on
    Returns:    

    """
    temp_shape = np.shape(template)
    temp_flat  = template.flatten("F")
    row0  = np.floor(indx[1]-temp_shape[1]/2)
    col0  = np.floor(indx[0]-temp_shape[0]/2)
    return target[int(col0):int(col0)+temp_shape[0]:1,int(row0):int(row0)+temp_shape[1]:1]

def custom_discretize(image,factor,mode=""):
    # Quantize to 16 levels of greyscale and downsample by 4 
    # (output image will have a 16-dim feature vec per pixel)
    if mode == "downsample":
        image = pixel_value_rescale(image,"uint4")
        dns = resize(image, (int(np.shape(image)[0]/factor),\
              int(np.shape(image)[1]/factor)), preserve_range=True).astype(np.uint8)
    else:
        print("Discretize mode {} not supported!".format(mode.upper()))
        dns = None
    return dns
    
def insert_image_patch(template,target,similarity,indx,path_to_save):
    """Generate coordinates of pixels within rectangle.
    Args:
        r0: uppermost row
        c0: leftmost column
    Returns:
        
    """
    temp_shape = np.shape(template)
    temp_flat  = template.flatten("F")
    row0  = np.floor(indx[1]-temp_shape[1]/2)
    col0  = np.floor(indx[0]-temp_shape[0]/2)
    rcrds = get_rectangle_crds(row0, col0, temp_shape[1], temp_shape[0])

    mod_img = target.copy()

    # Fill original image in with simulated patch 
    i = 0
    for entry in rcrds:
        mod_img[entry[1],entry[0]] = temp_flat[i]
        i += 1

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    axes[0, 0].imshow(target, cmap='gray')
    axes[0, 0].set_title('Experimental')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(template , cmap='gray')
    axes[0, 1].set_title('Simulated')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(target, cmap='gray')
    axes[1, 0].imshow(similarity, cmap='hot', alpha=0.4)
    axes[1, 0].set_title('Experimental (simulated similarity overlay)')
    axes[1, 0].plot(indx[1],indx[0],'*b',markersize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mod_img, cmap='gray')
    axes[1, 1].set_title('Simulated image placed inside experimental')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig("fit.png")
    plt.close()


def insert_image_patch_STM(template,target,similarity,indx,path_to_save):
    """Generate coordinates of pixels within rectangle.
    Args:
        r0: uppermost row
        c0: leftmost column
    Returns:
        
    """
    sct = MinMaxScaler(feature_range=(0, 255)).fit(template)
    template = sct.transform(template)

    temp_shape = np.shape(template)
    temp_flat  = template.flatten("F")
    row0  = np.floor(indx[1]-temp_shape[1]/2)
    col0  = np.floor(indx[0]-temp_shape[0]/2)
    rcrds = get_rectangle_crds(row0, col0, temp_shape[1], temp_shape[0])

    mod_img = target.copy()

    scm = MinMaxScaler(feature_range=(0, 255)).fit(mod_img)
    mod_img = scm.transform(mod_img)

    template = cv2.resize(template,(np.shape(target)[1],np.shape(target)[0]), interpolation=cv2.INTER_AREA)

    # Fill original image in with simulated patch 
    i = 0
    for entry in rcrds:
        mod_img[entry[1],entry[0]] = temp_flat[i]
        i += 1

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12.5, 5))

    axes[0].imshow(target, interpolation='quadric', cmap='hot')
    axes[0].set_title('STM Experiment',fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(template , interpolation='quadric', cmap='hot')
    axes[1].set_title('STM Simulation',fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(mod_img, interpolation='quadric', cmap='hot')
    axes[2].set_title('`ingrained` STM Simulation',fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save+"/fit.png")
    plt.close()

def score_vifp(im_true,im_test,sigma=2):
    # NOTE 0-255 scaling assumed!
    sc_true = MinMaxScaler(feature_range=(0, 255)).fit(im_true)
    sc_test = MinMaxScaler(feature_range=(0, 255)).fit(im_test)
    return 1 - vifp_mscale(sc_true.transform(im_true),sc_test.transform(im_test),sigma_nsq=sigma)