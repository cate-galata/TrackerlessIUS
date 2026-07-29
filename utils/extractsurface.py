import numpy as np
from skimage.measure import label  
from skimage.segmentation import find_boundaries 
from scipy.ndimage import binary_fill_holes
 
def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def get_corners(slice):
    pos_indices = np.where(slice)
    middle = int(pos_indices[0].mean())
    first_half =  np.where(slice[:middle,...])
    argmin_j1 = np.argmin(first_half[1])
    up_corner = first_half[0][argmin_j1], first_half[1][argmin_j1]
    
    sec_half =  np.where(slice[middle:,...])
    argmin_j2 = np.argmin(sec_half[1])
    low_corner = middle + sec_half[0][argmin_j2], sec_half[1][argmin_j2]

    dist_center = [np.abs(j - (up_corner[0]+low_corner[0])/2) for j in pos_indices[0]]
    u = np.argmin(dist_center)
    middle = pos_indices[0][u], pos_indices[1][u]
    
    return up_corner, low_corner, middle


def get_measurements(corner1, corner2, middle):
    a = np.array(corner1)
    b = np.array(middle) 
    c = np.array(corner2)
    
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)+1e-7)
    cosine_angle = np.min([cosine_angle, 0.999])
    cosine_angle = np.max([cosine_angle, -0.999])
    # print(cosine_angle, np.arccos(cosine_angle))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    dist_corner = np.linalg.norm(a-c)
    return angle, dist_corner



def test_reverse(data):
    return ((data[:,:data.shape[1]//2,data.shape[2]//2]==0).sum())<(data[:,data.shape[1]//2:,data.shape[2]//2]==0).sum()


def shift_image(img_data, k, return_trans=False, trans_x=None, trans_y=None):

    if trans_y is None or trans_x is None:
        pos = np.where(img_data[...,k]>0)

        mid_value_y = pos[1].mean()
        mid_value_x = pos[0].mean()
    
    if trans_y is None:
        trans_y = int(img_data.shape[1]/2-mid_value_y)
    if trans_x is None:
        trans_x = int(img_data.shape[0]/2-mid_value_x)
    
    diff_x = np.roll(img_data[...,k], trans_x, axis=0)
    diff_xy = np.roll(diff_x, trans_y, axis=1)
    if return_trans:
        return diff_xy, trans_x, trans_y
    else:
        return diff_xy
    

def filter_corners_inplace(input_array, corner_array, critical_values=[2,3]):
    """
    For slice z of corner_array:
    - Keep only points whose y coordinate is between the min and max y of critical values (2 and 3).
    - Everything else is set to 0.
    """
    for z in range(input_array.shape[-1]):
        slice_z = input_array[..., z]
        corner_z = corner_array[..., z]
        
        # Find coordinates of critical values
        ys, xs = np.where(np.isin(corner_z, critical_values))
        if len(ys) == 0:
            # No critical values, set everything to 0
            input_array[..., z] = 0
        else:
            # Compute y_min and y_max based only on critical values
            y_min = ys.min()
            y_max = ys.max()
            
            # Mask: keep only x between y_min and y_max
            mask = (np.arange(slice_z.shape[0])[:, None] >= y_min) & (np.arange(slice_z.shape[0])[:, None] <= y_max)
            
            # Apply mask: set values outside mask to 0
            input_array[..., z] = slice_z * mask