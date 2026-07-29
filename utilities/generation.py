from skimage.segmentation import find_boundaries 
from sklearn.decomposition import PCA
import SimpleITK as sitk
import os 
import nibabel as nib
from skimage.measure import label   
from skimage.segmentation import find_boundaries          
import numpy as np
from scipy.ndimage import binary_fill_holes
import pickle
from scipy.interpolate import RegularGridInterpolator


FLIPXY_44 = np.diag([-1, -1, 1, 1])

def normalize_dst(un_normal_point, ref):
    diff = un_normal_point - ref
    diff /= np.linalg.norm(diff)
    return ref + 5*diff

def find_third_point(corner_1,corner_2,d_c1_center,d_c2_center,max_ind):
    # Define points A, B and radii u, v
    A = corner_1[:2]  # Example: A(x_a, y_a)
    B = corner_2[:2]  # Example: B(x_b, y_b)
    u = d_c1_center  # Radius of circle centered at A
    v = d_c2_center  # Radius of circle centered at B

    # Compute d, the distance between A and B
    d = np.linalg.norm(B - A)

    # Compute midpoint M
    M = (A + B) / 2

    # Compute h, the distance from M to the intersection f
    h = np.sqrt(u**2 - (d / 2)**2)

    # Compute delta x and delta y
    delta_x = h * (B[1] - A[1]) / d
    delta_y = h * (B[0] - A[0]) / d

    # Calculate the intersection points
    C1 = M + np.array([delta_x, -delta_y])
    C2 = M + np.array([-delta_x, delta_y])
    
    C1 = np.round(C1,0)
    C2 = np.round(C2,0)
    
    # print(C1,C2)
    # if np.min(C1)<0:
    if corner_1[1]<max_ind//2:
        assert np.max(C2)<max_ind and np.min(C2)>0
        return C2.astype(np.uint32)
    else:
        assert np.max(C1)<max_ind and np.min(C1)>0
        return C1.astype(np.uint32)
    # else:
    #     raise "Ambiguous choice"
    

def get_tumor_landmarks(border, whole_tumor, core_tumor, affine_mr):
    points = dict()
    points_loc = dict()
    
    # Center of tumor core
    center = [k.mean() for k in np.where(core_tumor)]
    center += np.random.normal(loc=0.0, scale=3.0, size=(3))
    points_loc['center'] = center
    points['center'] = affine_mr.dot(center.tolist()+[1])[:3]
    
    # Main component tumor
    tumor_points = np.stack(np.where(whole_tumor),-1).astype(float)
    tumor_points -= center
    pca = PCA(n_components=1)
    pca.fit(tumor_points) 
    first_component = center + 10*pca.components_[0]
    points_loc['component'] = first_component
    points['component'] = affine_mr.dot(first_component.tolist()+[1])[:3]
    points['component'] = normalize_dst(points['component'],points['center'])
    
    # Contact 
    # border = find_boundaries(data_noncerebrum==1, mode='inner')
    border_pixels = np.stack(np.where(border),-1)
    nb_candidate = border_pixels.shape[0]
    
    border_pixels = np.stack(np.where(border),-1)
    ones = np.ones((nb_candidate, 1))

    # Stack the array of ones horizontally with the original array
    new_arr = np.hstack((border_pixels, ones))
    new_arr = np.dot(new_arr, affine_mr.T)[:,:3]
    
    prob_pixels = np.linalg.norm(new_arr - points['center'],axis=1)
    prob_pixels = np.exp(- prob_pixels)
    prob_pixels[prob_pixels<0] = 0
    prob_pixels /= prob_pixels.sum()
    selected_prob_center = np.random.choice(np.arange(nb_candidate), 1, p=prob_pixels)
    selected_prob_center = border_pixels[selected_prob_center,:].squeeze().tolist()
    points['contact'] = affine_mr.dot(selected_prob_center+[1])[:3]
    points_loc['contact'] = selected_prob_center
    
    return points, points_loc



def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def rigid_transform_3D(A, B, centroid_A=None, centroid_B=None):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    if centroid_A is None:
        centroid_A = np.mean(A, axis=1)
    if centroid_B is None:
        centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t =  -R @ centroid_A + centroid_B
    # t = centroid_B - centroid_A

    return R, t

def get_corners_us(data_us, data_maskus, affine_us):
    points = dict()
    points_loc = dict()
    
    # Mid-slice is selected 
    index_mid_us = data_us.shape[-1]//3
    index_mid_us = int(np.median(np.where(data_maskus==2)[2]))
    mask_2d = data_maskus[:,:,index_mid_us]
    
    # Get corners in this middle slice
    corner_1_ind = np.array(np.stack(np.where(mask_2d==2),-1)[0,:].tolist()+[index_mid_us])
    corner_2_ind = np.array(np.stack(np.where(mask_2d==3),-1)[0,:].tolist()+[index_mid_us])
    corner_1 = affine_us.dot(corner_1_ind.tolist()+[1])[:3]
    corner_2 = affine_us.dot(corner_2_ind.tolist()+[1])[:3]
    
    # Adding to the output dictionnaries
    points_loc['corner_1'] = corner_1_ind
    points_loc['corner_2'] = corner_2_ind
    points['corner_1'] = corner_1
    points['corner_2'] = corner_2
    
    return points, points_loc

    
def add_corners_mr(points_mr, points_mr_loc, points_us, border, affine_mr):
    border_points = np.stack(np.where(border),-1)
    ones = np.ones((border_points.shape[0], 1))

    # Stack the array of ones horizontally with the original array
    border_points_real = np.hstack((border_points, ones))
    border_points_real = np.dot(border_points_real, affine_mr.T)[:,:3]

    n_real = points_mr['component'] - points_mr['center']
    D_real = -np.dot(n_real, points_mr['contact'])

    dot_products_real = np.dot(border_points_real, n_real)

    # Check if each point satisfies the plane equation N.X + D = 0
    # We use a tolerance to account for floating-point errors
    tolerance = 2
    on_plane_real = np.abs(dot_products_real + D_real) < tolerance

    # Filter points that are on the plane
    points_on_plane_real = border_points_real[on_plane_real]
    points_on_plane = border_points[on_plane_real]

    segment_length = np.linalg.norm(points_us['corner_1']-points_us['corner_2'])

    dst_center = np.abs(np.linalg.norm(points_on_plane_real-points_mr['contact'],axis=1)-segment_length/2)
    corner_1_select = np.argmin(dst_center)
    corner_1_image_ind = points_on_plane[corner_1_select]
    corner_1_image_real = points_on_plane_real[corner_1_select]

    dst_corner_1_image = np.abs(np.linalg.norm(points_on_plane_real-corner_1_image_real,axis=1)-segment_length)
    corner_2_select = np.argmin(dst_center+dst_corner_1_image)
    corner_2_image_ind = points_on_plane[corner_2_select]
    corner_2_image_real = points_on_plane_real[corner_2_select]
    
    # Adding to the output dictionnaries
    points_mr_loc['corner_1'] = corner_1_image_ind
    points_mr_loc['corner_2'] = corner_2_image_ind
    points_mr['corner_1'] = corner_1_image_real
    points_mr['corner_2'] = corner_2_image_real
    return points_mr, points_mr_loc

def add_center_norm_us(points_us, points_us_loc, points_mr, affine_us, ratio_pixel=0.5, largest_dim=191):
    d_c1_center = np.linalg.norm(points_mr['corner_1']-points_mr['center'])
    d_c2_center = np.linalg.norm(points_mr['corner_2']-points_mr['center'])
    slice_index = points_us_loc['corner_1'][2]

    center_probe_ind = find_third_point(points_us_loc['corner_1'][:2], points_us_loc['corner_2'][:2], d_c1_center/ratio_pixel, d_c2_center/ratio_pixel, max_ind=largest_dim)
    
    points_us_loc['center'] =  center_probe_ind
    points_us['center'] = affine_us.dot(center_probe_ind.tolist()+[slice_index,1])[:3]
    points_us['component'] = affine_us.dot(center_probe_ind.tolist()+[slice_index-10,1])[:3]
    points_us['component'] = normalize_dst(points_us['component'], points_us['center'])
    return points_us, points_us_loc
    
    
    


def quantisize(img, levels=256, lower=0.0, upper=99.95):
    img_data = sitk.GetArrayFromImage(img).astype(np.float32)
    mask = img_data>0
    min_data = np.percentile(img_data[mask], lower)
    max_data = np.percentile(img_data[mask], upper)
    
    img_data[~mask] = min_data - 1
    img_data[img_data>max_data] = max_data
    img_data = (img_data-min_data) / (max_data - min_data+1e-8)
    img_data = np.digitize(img_data.squeeze(), np.arange(0,levels-1)/(levels-1)  ) 
    output = sitk.GetImageFromArray(img_data)
    output.CopyInformation(img)
    output =  sitk.Cast(output, sitk.sitkUInt8)
    return output

def _to_itk_convention(matrix):
    """RAS to LPS"""
    matrix = np.dot(FLIPXY_44, matrix)
    matrix = np.dot(matrix, FLIPXY_44)
    matrix = np.linalg.inv(matrix)
    return matrix

def _matrix_to_itk_transform(matrix, dimensions=3):
    matrix = _to_itk_convention(matrix)
    rotation = matrix[:dimensions, :dimensions].ravel().tolist()
    translation = matrix[:dimensions, 3].tolist()
    transform = sitk.AffineTransform(rotation, translation)
    return transform

    
def _create_itk_transform(matrix):
    """The tfm file contains the matrix from floating to reference."""
    transform = _matrix_to_itk_transform(matrix).GetInverse()
    return transform

def rigid_sitk(points_ref, dict_points_mov):
    R, t = rigid_transform_3D(dict_points_mov, points_ref)
    
    affine = np.zeros((4,4))
    affine[:3,:3] = R
    affine[:3,3:] = t
    affine[3,3] = 1
    affine_ras = np.linalg.inv(affine)
    transform = _create_itk_transform(affine_ras)
    return transform, R, t

def mask(img_to_mask, ref):
    img_data = sitk.GetArrayFromImage(img_to_mask).astype(np.float32)
    ref_data = sitk.GetArrayFromImage(ref).astype(np.float32)
    img_data*= (ref_data>0).astype(np.float32)
    output = sitk.GetImageFromArray(img_data)
    output.CopyInformation(img_to_mask)
    output =  sitk.Cast(output, sitk.sitkFloat32)
    return output  

def resample_seg(original_lab, target, transformation, labels=None):
    arrays = []
    labels = np.unique(sitk.GetArrayFromImage(original_lab).astype(np.uint8)).tolist()
    for i in labels:
        lab = sitk.GetImageFromArray((sitk.GetArrayFromImage(original_lab).astype(np.uint8)==i).astype(np.float32))
        lab.CopyInformation(original_lab)
        lab_resample = sitk.Resample(lab, target, transformation, sitk.sitkLinear)
        arrays.append(sitk.GetArrayFromImage(lab_resample))

    final_seg = np.argmax(np.stack(arrays,0),0).astype(np.uint8)
    # final_seg = np.stack(arrays,-1)
    final_seg = sitk.GetImageFromArray(final_seg)
    final_seg.CopyInformation(lab_resample)
    return final_seg 


def zero_mean(img):
    img_data = sitk.GetArrayFromImage(img).astype(np.float32)
    img_data = img_data - img_data.min()
    output = sitk.GetImageFromArray(img_data)
    output.CopyInformation(img)
    output =  sitk.Cast(output, sitk.sitkInt16)
    return output

def zeros_like(ref):
    img_data = np.zeros_like(sitk.GetArrayFromImage(ref).astype(np.uint8))
    output = sitk.GetImageFromArray(img_data)
    output.CopyInformation(ref)
    output =  sitk.Cast(output, sitk.sitkFloat32)
    return output

def select_ref_data(path_folder,case):
    path_us_intraop = os.path.join(path_folder, case, 'Intraop-US')
    preduras = [k for k in os.listdir(path_us_intraop) if '.nrrd' in k and 'pre_dura' in k]
    if len(preduras)>0: # Existing pre-dura US
        path_predura_scan = os.path.join(path_us_intraop, preduras[0])
        path_premr = os.path.join(path_folder, case, 'Preop-MR')
        mr = [k for k in os.listdir(path_premr) if '.nrrd' in k]
        mr3d = [k for k in os.listdir(path_premr) if '3d' in k.lower()]
        mr2d = [k for k in os.listdir(path_premr) if '3d' in k.lower()]
        
        path_mr_scan = None
        if len(mr3d)>0: # First we pick 3D scans
            space3d = [k for k in mr3d if 'space' in k.lower()]
            t23d = [k for k in mr3d if 't2' in k.lower() and not 'flair' in k.lower()]
            flair3d = [k for k in mr3d if 'flair' in k.lower()]
            t1c3d = [k for k in mr3d if 'postcontrast' in k.lower()]
            others3d = [k for k in mr3d if not k in space3d+t23d+flair3d+t1c3d]
            for set in [space3d, t23d, flair3d, t1c3d, others3d]:
                if len(set)>0 and path_mr_scan is None:
                    res_setscans = {
                        k: np.mean(sitk.ReadImage(os.path.join(path_premr,k)).GetSpacing()) for k in set
                        }
                    path_mr_scan = min(res_setscans, key=res_setscans.get)
                    path_mr_scan = os.path.join(path_premr, path_mr_scan)

        elif len(mr2d)>0: # Second we pick 2D scans
            space2d = [k for k in mr2d if 'space' in k.lower()]
            t22d = [k for k in mr2d if 't2' in k.lower() and not 'flair' in k.lower()]
            flair2d = [k for k in mr2d if 'flair' in k.lower()]
            t1c2d = [k for k in mr2d if 'postcontrast' in k.lower()]
            others2d = [k for k in mr2d if not k in space3d+t23d+flair3d+t1c3d]
            for set in [space2d, t22d, flair2d, t1c2d, others2d]:
                if len(set)>0 and path_mr_scan is None:
                    res_setscans = {
                        k: np.mean(sitk.ReadImage(os.path.join(path_premr,k)).GetSpacing()) for k in set
                        }
                    path_mr_scan = min(res_setscans, key=res_setscans.get)
                    path_mr_scan = os.path.join(path_premr, path_mr_scan)
        if path_mr_scan is None:
            print(f'error with {case}')
        else:
            if case=='Case026':
                path_mr_scan = f'{path_folder}/Case026/Preop-MR/Case026-preop-MR-3D_AX_T1_postcontrast.nrrd'
            path_imgs = {'us':path_predura_scan, 'mr':path_mr_scan}
        return path_imgs


def select_highest(list_imgs):
    if len(list_imgs)>0:
        res_setscans = {k: np.mean(sitk.ReadImage(k).GetSpacing()) for k in list_imgs } 
        return  [min(res_setscans, key=res_setscans.get)]
    else:
        return []
    
def get_mr_images(path_folder, case, threed_only=False):
    path_premr = os.path.join(path_folder, case, 'Preop-MR')
    
    if threed_only:
        T2s = [os.path.join(path_premr,k) for k in os.listdir(path_premr) if '.nrrd' in k if 't2' in k.lower() and not 'flair' in k.lower() and 'space' in k.lower()]
        flairs = [os.path.join(path_premr,k) for k in os.listdir(path_premr) if '.nrrd' in k if 'flair' in k.lower() and '3d' in k.lower()]
        T1ce = [os.path.join(path_premr,k) for k in os.listdir(path_premr) if '.nrrd' in k if 'postcontrast' in k.lower() and '3d' in k.lower()]
    else:
        T2s = [os.path.join(path_premr,k) for k in os.listdir(path_premr) if '.nrrd' in k if 't2' in k.lower() and not 'flair' in k.lower()]
        flairs = [os.path.join(path_premr,k) for k in os.listdir(path_premr) if '.nrrd' in k if 'flair' in k.lower()]
        T1ce = [os.path.join(path_premr,k) for k in os.listdir(path_premr) if '.nrrd' in k if 'postcontrast' in k.lower() ]        
        
    T2s = select_highest(T2s)
    flairs = select_highest(flairs)
    T1ce = select_highest(T1ce)
    
    T2s = [('t2',k) for k in T2s]
    flairs = [('flair',k) for k in flairs]
    T1ce = [('cet1',k) for k in T1ce]
    # T1pre = [('pret1',k) for k in T1pre]

    mrs_to_register = T2s + T1ce + flairs 
    return mrs_to_register


def get_coregistered_mr_images(path_folder, case, threed_only=False):
    path_premr = os.path.join(path_folder, case)
    
    if threed_only:
        T2s = [os.path.join(path_premr,k) for k in os.listdir(path_premr) if '.nrrd' in k if 't2' in k.lower() and not 'flair' in k.lower() and 'space' in k.lower()]
        flairs = [os.path.join(path_premr,k) for k in os.listdir(path_premr) if '.nrrd' in k if 'flair' in k.lower() and '3d' in k.lower()]
        T1ce = [os.path.join(path_premr,k) for k in os.listdir(path_premr) if '.nrrd' in k if 'postcontrast' in k.lower() and '3d' in k.lower()]
    else:
        T2s = [path_premr for k in os.listdir(path_premr) if '.nii.gz' in k if 't2' in k.lower()]
        flairs = [path_premr for k in os.listdir(path_premr) if '.nii.gz' in k if 'flair' in k.lower()]
        T1ce = [path_premr for k in os.listdir(path_premr) if '.nii.gz' in k if 'cet1' in k.lower()]        
    
    T2s = [('t2',k) for k in T2s]
    flairs = [('flair',k) for k in flairs]
    T1ce = [('cet1',k) for k in T1ce]

    mrs_to_register = T2s + T1ce + flairs 
    return mrs_to_register


def save_all(items):
    """Debug helper: saves each element of `items` as ./experiments/{i}.pickle."""
    for i, element in enumerate(items):
        with open(f"./experiments/{i}.pickle", "wb") as handle:
            pickle.dump(element, handle, protocol=pickle.HIGHEST_PROTOCOL)



def find_plane_equation(p1, p2, p3):
    # Calculate vectors v1 and v2 in the plane
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Compute the normal vector to the plane (n = v1 x v2)
    n = np.cross(v1, v2)
    
    # The plane equation is n[0]*x + n[1]*y + n[2]*z + D = 0
    # Solve for D using one of the points (p1, p2, or p3)
    D = -np.dot(n, p1)
    
    return n, D

def z_value_for_xy(x, y, n, D):
    # Assuming the plane equation is n[0]*x + n[1]*y + n[2]*z + D = 0
    # Solve for z
    z = (-D - n[0]*x - n[1]*y) / n[2]
    return z

def z_value_for_xy_grid(x_grid, y_grid, n, D):
    # Calculate z for each (x, y) pair in the grid
    z_grid = (-D - n[0] * x_grid - n[1] * y_grid) / n[2]
    return z_grid


def create_reslicing(path_dataset, path_mask, flnm):
    img = nib.load(os.path.join(path_dataset, flnm))
    data = img.get_fdata()
    affine = img.affine
    case_us = flnm.split('-')[1]
    mask = nib.load(path_mask.format(case_us)).get_fdata()
    x,y,z = np.where(mask==2)
    point_1 = np.array([x[np.argmin(z)], y[np.argmin(z)], z[np.argmin(z)]])
    x,y,z = np.where(mask==3)
    point_2 = np.array([x[np.argmax(z)], y[np.argmin(z)], z[np.argmin(z)]])
    
    good_orientation = np.median(y)<191//2

    start = point_1[2]

    slices_img = []
    for z_index in range(start):
        # Find the plane equation coefficients
        if good_orientation:
            point_3 = np.array([191//2,191,z_index])
        else:
            point_3 = np.array([191//2,0,z_index])
        n, D = find_plane_equation(point_1, point_2, point_3)

        # Generate a grid of x and y values
        x_values = np.linspace(0, 191, 192)  # 193 points from 0 to 192 inclusive
        y_values = np.linspace(0, 191, 192)
        x_grid, y_grid = np.meshgrid(x_values, y_values)

        # Calculate the z values for the grid
        z_grid = z_value_for_xy_grid(x_grid, y_grid, n, D)
        z = np.linspace(0, data.shape[-1]-1, data.shape[-1])  # Z coordinates of the volume data

        # Create the interpolator object
        interpolator = RegularGridInterpolator((x_values, y_values, z), data, bounds_error=False, fill_value=0)

        # Flatten the grids to create a list of points for interpolation
        points = np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()]).T

        # Perform the interpolation
        interpolated_values = interpolator(points)

        # Reshape the interpolated values back to the 2D grid shape
        slice_image = interpolated_values.reshape(x_grid.shape)
        slices_img.append(np.round(slice_image.T,0).astype(np.uint8))
        
    x,y,z = np.where(mask==2)
    point_1 = np.array([x[np.argmax(z)], y[np.argmax(z)], z[np.argmax(z)]])
    x,y,z = np.where(mask==3)
    point_2 = np.array([x[np.argmax(z)], y[np.argmax(z)], z[np.argmax(z)]])
    end = point_1[2]
    for z_index in range(start, end):
        slices_img.append(data[:,:,z_index])
    
    for z_index in range(end,data.shape[-1]):
        # Find the plane equation coefficients
        if good_orientation:
            point_3 = np.array([191//2,191,z_index])
        else:
            point_3 = np.array([191//2,0,z_index])
        n, D = find_plane_equation(point_1, point_2, point_3)

        # Generate a grid of x and y values
        x_values = np.linspace(0, 191, 192)  # 193 points from 0 to 192 inclusive
        y_values = np.linspace(0, 191, 192)
        x_grid, y_grid = np.meshgrid(x_values, y_values)

        # Calculate the z values for the grid
        z_grid = z_value_for_xy_grid(x_grid, y_grid, n, D)
        z = np.linspace(0, data.shape[-1]-1, data.shape[-1])  # Z coordinates of the volume data

        # Create the interpolator object
        interpolator = RegularGridInterpolator((x_values, y_values, z), data, bounds_error=False, fill_value=0)

        # Flatten the grids to create a list of points for interpolation
        points = np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()]).T

        # Perform the interpolation
        interpolated_values = interpolator(points)

        # Reshape the interpolated values back to the 2D grid shape
        slice_image = interpolated_values.reshape(x_grid.shape)
        
        slices_img.append(np.round(slice_image.T,0).astype(np.uint8))
        
    return nib.Nifti1Image(np.stack(slices_img,-1), affine)

    # nib.Nifti1Image(np.stack(slices_img,-1), affine).to_filename(f'./data/synthetic/bratsv2/reslice/reslice{case}.nii.gz')
    # nib.Nifti1Image(np.stack(slices_seg,-1), affine).to_filename(f'./data/synthetic/bratsv2/reslice/{case}-seg.nii.gz')
    # nib.Nifti1Image(np.stack(slices_calibration,-1), affine).to_filename(f'./data/synthetic/bratsv2/reslice/{case}-tracking.nii.gz')


    
    
    