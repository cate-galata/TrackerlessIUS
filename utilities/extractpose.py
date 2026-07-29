import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
from utilities.extractsurface import * 

def flip_point(pt, width):
    flipped_x = (width - 1) - pt[0]
    return (flipped_x, pt[1], pt[2])


def transform_pts_us_to_mri(pts_us, us_img, mri_img, sitk_transform, output_path, case):

    pts_mri = []
    pts_mri_world = []

    for z, y, x in pts_us:

        # Convert NumPy index -> SimpleITK index
        physical = us_img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))

        # Transform into MRI physical coordinates
        pt_mri = sitk_transform.TransformPoint(physical)
        pts_mri_world.append(pt_mri)

        # Convert MRI physical -> MRI index
        idx = mri_img.TransformPhysicalPointToIndex(pt_mri)   # (x,y,z)

        # Save if inside bounds
        if (
            0 <= idx[0] < mri_img.GetWidth()  and
            0 <= idx[1] < mri_img.GetHeight() and
            0 <= idx[2] < mri_img.GetDepth()
        ):
            pts_mri.append(idx)

        pts_mri_np = np.zeros(sitk.GetArrayFromImage(mri_img).shape, dtype=np.uint8)

        for x, y, z in pt_mri:
            pts_mri_np[z, y, x] = 1     # NumPy = (z,y,x)

        pts_mri = sitk.GetImageFromArray(pts_mri_np)
        pts_mri.CopyInformation(mri_img)   # origin, spacing, direction


        output_path_folder = os.path.join(output_path, case)
        os.makedirs(output_path_folder, exist_ok=True)
        output_path_file = os.path.join(output_path_folder, f'{case}-P.nii.gz')
        sitk.WriteImage(
            image=pts_mri, 
            fileName=os.path.join(output_path_file),
            useCompression=True
        )

    return pts_mri

def trajectory_direction_from_points(traj_points):
    """
    traj_points: (N,3) points in world coordinates (float)
    returns:
      - dir_unit: (3,) unit vector giving primary direction of the trajectory
      - centroid: (3,) centroid of the trajectory points
      - traj_length: float, total Euclidean path length (sum of segment lengths)
    """
    traj_points = np.asarray(traj_points, dtype=float)
    if traj_points.shape[0] < 2:
        raise ValueError("Need at least 2 points to define a trajectory direction.")
    centroid = traj_points.mean(axis=0)
    # center and do SVD: principal direction = first right-singular vector
    centered = traj_points - centroid
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    dir_unit = Vt[0]  # already unit length
    # ensure unit
    dir_unit = dir_unit / np.linalg.norm(dir_unit)
    # compute simple path length (sum of segment lengths)
    segs = np.linalg.norm(np.diff(traj_points, axis=0), axis=1)
    traj_length = segs.sum()
    return dir_unit, centroid, traj_length

def alignment_with_pcs(traj_dir, tumor_pcs, return_details=True, oriented=False):
    """
    traj_dir: (3,) unit vector (trajectory direction)
    tumor_pcs: (3,3) matrix whose columns are eigenvectors (pc1,pc2,pc3) in world coords
    oriented: if True, do not take absolute value of cosine (keeps sign)
    returns:
      - best_idx: index (0,1,2) of most aligned PC
      - cosines: array(3,) absolute cosines (or signed if oriented=True)
      - angles_deg: array(3,) corresponding angles in degrees
    """
    traj_dir = np.asarray(traj_dir, dtype=float)
    pcs = np.asarray(tumor_pcs, dtype=float)
    # Ensure unit length for safety
    traj_dir = traj_dir / np.linalg.norm(traj_dir)
    pcs_norm = pcs / np.linalg.norm(pcs, axis=0, keepdims=True)
    # dot with each PC (columns)
    dots = pcs_norm.T.dot(traj_dir)  # shape (3,)
    if not oriented:
        cosines = np.abs(dots)
    else:
        cosines = dots
    # clip numerical drift
    cosines_clipped = np.clip(cosines, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cosines_clipped))
    best_idx = int(np.argmax(np.abs(cosines)))  # choose max absolute cosine if oriented=False
    if return_details:
        return best_idx, cosines, angles_deg
    return best_idx

def projection_span_along_axis(traj_points, axis):
    """
    Gives the span (max - min) of projection values of traj_points onto axis.
    Useful to know how much of the trajectory lies along that axis (in mm).
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    proj_vals = np.dot(traj_points, axis)
    span = proj_vals.max() - proj_vals.min()
    return span

def precompute_Ps(img_us, data_us, tfm, Z_MID, Z_MAX, left_corners, corners_corrected, P_corrected, i_vec_corrected, reverse_needed):
    
    # --- Start at Z_MID ---
    points_mask_mid = left_corners[..., Z_MID]
    if np.any(points_mask_mid):
        ys, xs = np.where(points_mask_mid > 0)
        labels = points_mask_mid[ys, xs]
        
        coords = {}
        for value in [2, 3]:
            idx = labels == value
            x_mean = int(xs[idx][0])
            y_mean = int(ys[idx][0])
            coords[value] = (y_mean, x_mean, Z_MID)

        left_top_us = flip_point(coords[2], data_us.shape[1]) if reverse_needed else coords[2]
        left_bottom_us = flip_point(coords[3], data_us.shape[1]) if reverse_needed else coords[3]
            
        left_top_world = img_us.TransformIndexToPhysicalPoint(left_top_us)
        left_top_mri = tfm.TransformPoint(left_top_world)
        left_bottom_world = img_us.TransformIndexToPhysicalPoint(left_bottom_us)
        left_bottom_mri = tfm.TransformPoint(left_bottom_world)
        
        P_mid = (np.array(left_top_mri) + np.array(left_bottom_mri)) / 2
        i_vec = np.array(left_top_mri) - np.array(left_bottom_mri)
        i_vec_unit = i_vec / np.linalg.norm(i_vec)

        P_corrected[Z_MID, :] = P_mid
        i_vec_corrected[Z_MID, :] = i_vec_unit
    else:
        # Handle error case: if the middle slice is also empty, stop.
        print("ERROR: Mid-slice is empty. Cannot initialize P.")
        return

    # --- B. PROPAGATE BACKWARDS (Z_MID - 1 down to 0) ---
    for z in range(Z_MID - 1, -1, -1):
        
        current_corners = corners_corrected[..., z]
        
        if np.any(current_corners):
            # Valid data: Recalculate P using corrected corners
            ys, xs = np.where(current_corners > 0)
            labels = current_corners[ys, xs]
            
            coords = {}
            for value in [2, 3]:
                idx = labels == value
                x_mean = int(xs[idx][0])
                y_mean = int(ys[idx][0])
                coords[value] = (y_mean, x_mean, Z_MID)

            left_top_us = flip_point(coords[2], data_us.shape[1]) if reverse_needed else coords[2]
            left_bottom_us = flip_point(coords[3], data_us.shape[1]) if reverse_needed else coords[3]

            left_top_world = img_us.TransformIndexToPhysicalPoint(left_top_us)
            left_top_mri = tfm.TransformPoint(left_top_world)
            left_bottom_world = img_us.TransformIndexToPhysicalPoint(left_bottom_us)
            left_bottom_mri = tfm.TransformPoint(left_bottom_world)
                
            P_new = (np.array(left_top_mri) + np.array(left_bottom_mri)) / 2
            i_vec_new = np.array(left_top_mri) - np.array(left_bottom_mri)
            i_vec_unit_new = i_vec_new / np.linalg.norm(i_vec_new)

            P_corrected[z, :] = P_new
            i_vec_corrected[z, :] = i_vec_unit_new
        else:
            # Invalid data: Use P from the *next* slice (z+1)
            # and update the Z coordinate to the current slice z
            P_from_next = P_corrected[z + 1, :].copy()
            # P_from_next[2] = z # Set Z coordinate to current slice
            P_corrected[z, :] = P_from_next

            i_from_next = i_vec_corrected[z + 1, :].copy()
            i_from_next[2] = z
            i_vec_corrected[z, :] = i_from_next
            
    # --- C. PROPAGATE FORWARDS (Z_MID + 1 up to Z_MAX - 1) ---
    for z in range(Z_MID + 1, Z_MAX):
        
        current_corners = corners_corrected[..., z]
        
        if np.any(current_corners):
            # Valid data: Recalculate P
            ys, xs = np.where(current_corners > 0)
            labels = current_corners[ys, xs]
            
            coords = {}
            for value in [2, 3]:
                idx = labels == value
                x_mean = int(xs[idx][0])
                y_mean = int(ys[idx][0])
                coords[value] = (y_mean, x_mean, Z_MID)

            left_top_us = flip_point(coords[2], data_us.shape[1]) if reverse_needed else coords[2]
            left_bottom_us = flip_point(coords[3], data_us.shape[1]) if reverse_needed else coords[3]
                
            left_top_world = img_us.TransformIndexToPhysicalPoint(left_top_us)
            left_top_mri = tfm.TransformPoint(left_top_world)
            left_bottom_world = img_us.TransformIndexToPhysicalPoint(left_bottom_us)
            left_bottom_mri = tfm.TransformPoint(left_bottom_world)
                
            P_new = (np.array(left_top_mri) + np.array(left_bottom_mri)) / 2
            i_vec_new = np.array(left_top_mri) - np.array(left_bottom_mri)
            i_vec_unit_new = i_vec_new / np.linalg.norm(i_vec_new)

            P_corrected[z, :] = P_new
            i_vec_corrected[z, :] = i_vec_unit_new
        else:
            # Invalid data: Use P from the *previous* slice (z-1)
            # and update the Z coordinate to the current slice z
            P_from_prev = P_corrected[z - 1, :].copy()
            # P_from_prev[2] = z # Set Z coordinate to current slice
            P_corrected[z, :] = P_from_prev

            i_from_prev = i_vec_corrected[z - 1, :].copy()
            i_from_prev[2] = z
            i_vec_corrected[z, :] = i_from_prev

    return P_corrected, i_vec_corrected


def slice_pose_nib_torch(
    img: nib.spatialimages.SpatialImage,
    point_world,
    dir_world,
    inplane_world,
    dx_mm: float,
    dy_mm: float,
    H_out: int,
    W_out: int,
    pad_value: float = 0.0,
    return_world_coords: bool = False,
):
    """
    Sample a 2D slice from a 3D nibabel image using PyTorch grid_sample,
    given a pose in WORLD space.

    Geometry / contract
    --------------------
    The slice is defined by:
      - A 3D point P in world coordinates (`point_world`).
      - A direction d in world coordinates (`dir_world`) for the slice X-axis.
      - A direction i in world coordinates (`inplane_world`) for the slice Y-axis.

    The mapping from slice pixel (y, x) to world coordinates is:

        world(y, x) = P
                      + (y - H_out/2) * dy_mm * y_hat
                      + x            * dx_mm * x_hat

    where:
      - x_hat = dir_world normalized
      - y_hat = inplane_world normalized
      - P = point_world

    So:
      - Pixel (y = H_out/2, x = 0) corresponds exactly to point_world.
      - The slice X-axis (columns) is colinear with dir_world.
      - The slice Y-axis (rows) is colinear with inplane_world.
      - dir_world and inplane_world must be non-colinear (otherwise the
        slice plane is not well-defined).
      - Pixel spacing is:
          * dx_mm along X (columns, dir_world direction),
          * dy_mm along Y (rows, inplane_world direction).
      - The basis (x_hat, y_hat) is not forced to be orthogonal; the slice
        can be skewed in 3D.

    Implementation details
    ----------------------
    - The 3D image data is taken as `data = img.get_fdata()` with shape
      (nx, ny, nz) = (i, j, k).
    - World coordinates are defined by the affine:
         world = affine @ [i, j, k, 1]^T
    - We build world coords for each slice pixel, convert them back to voxel
      indices (i,j,k), then rearrange the volume to (D,H,W) = (z,y,x) and
      use torch.nn.functional.grid_sample in 3D:
         input : (N, C, D, H, W)
         grid  : (N, D_out, H_out, W_out, 3) with (x_norm, y_norm, z_norm)

    Parameters
    ----------
    img : nib.spatialimages.SpatialImage
        3D image with affine (e.g., nib.Nifti1Image).
        Data is assumed to be in array space with shape (nx, ny, nz).
    point_world : array-like, shape (3,)
        3D point P in world coordinates (mm). This will appear at pixel
        (y = H_out/2, x = 0) in the slice.
    dir_world : array-like, shape (3,)
        Direction in world coordinates (mm) for the slice X-axis (columns).
        Only direction matters; magnitude is ignored.
    inplane_world : array-like, shape (3,)
        Direction in world coordinates (mm) for the slice Y-axis (rows).
        Only direction matters; magnitude is ignored. Must not be colinear
        with dir_world.
    dx_mm : float
        Pixel spacing along the slice X-axis (columns) in mm.
    dy_mm : float
        Pixel spacing along the slice Y-axis (rows) in mm.
    H_out, W_out : int
        Output slice height (rows) and width (cols).
    pad_value : float, optional
        Value to use for out-of-volume pixels. grid_sample itself pads with
        zeros; if pad_value != 0, we post-process using a mask. Default 0.0.
    return_world_coords : bool, optional
        If True, also return the world coordinates of each pixel center
        as an array of shape (H_out, W_out, 3).

    Returns
    -------
    slice_2d : ndarray, shape (H_out, W_out)
        Sampled slice as a NumPy array.
    pts_world : ndarray, shape (H_out, W_out, 3), optional
        Only returned if return_world_coords=True.
        World coordinates (mm) of each pixel center.
    """
    # --- 0) Data and affine ---
    data = img.get_fdata().astype(np.float32)   # (nx, ny, nz) = (i, j, k)
    affine = img.affine
    nx, ny, nz = data.shape

    # --- 1) Normalize directions in world space ---
    p = np.asarray(point_world, dtype=np.float32)
    d = np.asarray(dir_world, dtype=np.float32)
    i_vec = np.asarray(inplane_world, dtype=np.float32)

    # X-axis: along dir_world
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-6:
        raise ValueError("dir_world has near-zero magnitude.")
    x_hat = d / d_norm

    # Y-axis: along inplane_world
    i_norm = np.linalg.norm(i_vec)
    if i_norm < 1e-6:
        raise ValueError("inplane_world has near-zero magnitude.")
    y_hat = i_vec / i_norm

    # Check they are not colinear (no degenerate plane)
    cross_xy = np.cross(x_hat, y_hat)
    if np.linalg.norm(cross_xy) < 1e-6:
        raise ValueError(
            "dir_world and inplane_world are (nearly) colinear; "
            "they must span a 2D plane."
        )

    # --- 2) Slice pixel grid (rows = y, cols = x) ---
    ys = np.arange(H_out, dtype=np.float32)   # 0..H_out-1
    xs = np.arange(W_out, dtype=np.float32)   # 0..W_out-1

    # P must be at (y0, x=0), with y0 = H_out/2
    y0 = H_out / 2.0

    Y, X = np.meshgrid(ys, xs, indexing="ij")  # (H_out, W_out)

    # Offsets in world space
    offset_y = (Y - y0) * dy_mm   # along y_hat
    offset_x = X * dx_mm          # along x_hat

    pts_world = (
        p[None, None, :]
        + offset_y[..., None] * y_hat[None, None, :]
        + offset_x[..., None] * x_hat[None, None, :]
    )  # (H_out, W_out, 3)

    # --- 3) World -> voxel indices (i,j,k) ---
    pts_world_flat = pts_world.reshape(-1, 3)
    inv_aff = np.linalg.inv(affine)
    ijk_flat = apply_affine(inv_aff, pts_world_flat)  # (N,3)
    ijk = ijk_flat.reshape(H_out, W_out, 3)

    i_idx = ijk[..., 0]  # axis 0 in data
    j_idx = ijk[..., 1]  # axis 1
    k_idx = ijk[..., 2]  # axis 2

    # --- 4) Prepare volume for grid_sample: (N,C,D,H,W) = (1,1,z,y,x) ---
    # data is (nx, ny, nz) = (i, j, k)
    # we reorder to (k, j, i) = (z, y, x)
    vol_np = np.transpose(data, (2, 1, 0))  # (nz, ny, nx) = (D,H,W)
    D, H, W = vol_np.shape

    vol_t = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

    # --- 5) Build normalized grid for grid_sample ---
    # Our voxel indices (i,j,k) map to (x,y,z) in the reordered volume:
    # x_index -> i_idx, y_index -> j_idx, z_index -> k_idx
    X_idx = i_idx
    Y_idx = j_idx
    Z_idx = k_idx

    X_t = torch.from_numpy(X_idx.astype(np.float32))
    Y_t = torch.from_numpy(Y_idx.astype(np.float32))
    Z_t = torch.from_numpy(Z_idx.astype(np.float32))

    # Convert voxel indices to normalized coordinates in [-1,1]
    # with align_corners=True:
    x_norm = 2.0 * X_t / (W - 1) - 1.0
    y_norm = 2.0 * Y_t / (H - 1) - 1.0
    z_norm = 2.0 * Z_t / (D - 1) - 1.0

    grid = torch.stack((x_norm, y_norm, z_norm), dim=-1)  # (H_out, W_out, 3)
    grid = grid.unsqueeze(0).unsqueeze(1)                 # (1,1,H_out,W_out,3)

    vol_t = vol_t.to(grid.dtype)

    # --- 6) Sample with grid_sample ---
    slice_t = F.grid_sample(
        vol_t,
        grid,
        mode="bilinear",
        padding_mode="zeros",   # zeros outside [-1,1]
        align_corners=True,
    )  # (1,1,1,H_out,W_out)

    slice_2d = slice_t[0, 0, 0].cpu().numpy()  # (H_out, W_out)

    # If user wants a non-zero pad_value, mask out-of-bounds explicitly
    if pad_value != 0.0:
        mask_oob = (
            (X_idx < 0) | (X_idx > nx - 1) |
            (Y_idx < 0) | (Y_idx > ny - 1) |
            (Z_idx < 0) | (Z_idx > nz - 1)
        )
        slice_2d[mask_oob] = pad_value

    if return_world_coords:
        return slice_2d, pts_world
    else:
        return slice_2d


def visualize_slice_with_pose(
    slice_2d,
    P_world,
    dir_world,
    inplane_world,
    dx_mm,
    dy_mm,
    H_out,
    W_out,
    case, 
    z_vis,
    output_folder,
    length_mm=30,
    flip_y=True,
):
    """
    Visualize the slice with P, x-axis (dir_world), and y-axis (inplane_world).

    Parameters
    ----------
    slice_2d : (H_out, W_out) numpy array
        The slice produced by slice_pose_nib_torch.
    P_world : (3,) array
        World coordinates of the point P.
        *This is NOT used to plot unless you want 3D.*
        Its 2D pixel location is always (H_out/2, 0).
    dir_world : (3,) array
        Direction vector for the slice X-axis (columns).
    inplane_world : (3,) array
        Direction vector for the slice Y-axis (rows).
    dx_mm, dy_mm : float
        Pixel spacing in mm (columns, rows).
    H_out, W_out : int
    length_mm : float
        How long to draw the axes arrows.
    flip_y : bool
        If True: flips the image vertically for correct display.
    """

    # Pixel location of P
    y0 = H_out / 2
    x0 = 0

    # Convert length in mm to approximate pixel length
    dx_pix = length_mm / dx_mm      # axis length in x-pixels
    dy_pix = length_mm / dy_mm      # axis length in y-pixels

    # Normalize slice basis vectors
    d_hat = dir_world / np.linalg.norm(dir_world)
    i_hat = inplane_world / np.linalg.norm(inplane_world)

    # In slice *pixel* coordinates:
    # x-axis (columns) → (dx_pix, 0)
    # y-axis (rows)    → (0,    dy_pix)

    # Prepare image for display
    img_disp = slice_2d[::-1, :] if flip_y else slice_2d

    # Flip P pixel y coordinate if necessary
    if flip_y:
        y0_disp = H_out - y0
    else:
        y0_disp = y0

    plt.figure(figsize=(6, 6))
    plt.imshow(img_disp, cmap='gray', origin='upper')
    plt.axis('off')

    # Draw point P
    plt.scatter([x0], [y0_disp], c='red', s=80, label="P")

    # Draw x-axis arrow (dir_world)
    plt.arrow(
        x0,
        y0_disp,
        dx_pix,
        0,
        color='orange',
        width=0.5,
        length_includes_head=True
    )
    plt.text(x0 + dx_pix * 1.1, y0_disp, "x (dir_world)", color='orange')

    # Draw y-axis arrow (inplane_world)
    plt.arrow(
        x0,
        y0_disp,
        0,
        -dy_pix if flip_y else dy_pix,
        color='cyan',
        width=0.5,
        length_includes_head=True
    )
    plt.text(x0, y0_disp - dy_pix * (1.3 if flip_y else -1.3),
             "y (inplane_world)", color='cyan')
    
    output_folder = f'{output_folder}/{case}'
    os.makedirs(output_folder, exist_ok=True)

    plt.savefig(f'{output_folder}/sliced_mri_{z_vis}.png')
    plt.close()


def extract_us_pose_keypoints(data):
    z_mid = data.shape[-1] //2 
    
    # Reverse ultrasound if needed (probe surface should be on the right side)
    reverse_needed = test_reverse(data)
    if reverse_needed:
        data = data[:,::-1,:]

    # First, we can extract the border of the field-of-view of the iUS.
    mask = getLargestCC(binary_fill_holes((data>0).astype(np.uint8)))
    bound = find_boundaries(mask, mode='inner')
    
    # Second, we identify the left border and center of the field-of-view.
    left_contact = np.zeros_like(mask,dtype=np.uint8)
    right_contact = np.zeros_like(mask,dtype=np.uint8)

    for z in range(data.shape[-1]):
        bound = find_boundaries(mask[..., z], mode='inner')
        y_indices, x_indices = np.where(bound)
        for y in np.unique(y_indices):
            x_min = np.min(x_indices[y_indices == y])
            left_contact[y, x_min, z] = 1
            x_max = np.max(x_indices[y_indices == y])
            right_contact[y, x_max, z] = 1
    
    # Third, we identify the top and bottom corners.
    corners = np.zeros_like(left_contact)
    for z in range(data.shape[-1]):
        if np.sum(left_contact[...,z])>10:
            data_shifted, trans_x, trans_y = shift_image(left_contact, z, trans_y=0, return_trans=True)
            
            top_corner, bottom_corner, mid_point = get_corners(data_shifted)

            P = ((top_corner[0] + bottom_corner[0]) // 2, (top_corner[1] + bottom_corner[1]) // 2)

            corners[mid_point[0]-trans_x, mid_point[1]-trans_y, z] = 1
            corners[top_corner[0]-trans_x, top_corner[1]-trans_y, z] = 2
            corners[bottom_corner[0]-trans_x,  bottom_corner[1]-trans_y, z] = 3
            corners[P[0]-trans_x, P[1]-trans_y, z] = 4

    # Fourth, we remove the corners that don't match the probe geometry based on some heuristics
    data_shifted_mid, trans_x_mid, trans_y_mid = shift_image(left_contact, z_mid, trans_y=0, return_trans=True)
    top_corner_mid, bottom_corner_mid, mid_point_mid = get_corners(data_shifted_mid)
    ref_angle, ref_dist = get_measurements(top_corner_mid, bottom_corner_mid, mid_point_mid)
    
    data_shifted_mid, trans_x_mid, trans_y_mid = shift_image(right_contact, z_mid, trans_y=0, return_trans=True)
    top_corner_mid, bottom_corner_mid, mid_point_mid = get_corners(data_shifted_mid)
    ref_angle_right, ref_dist_right = get_measurements(top_corner_mid, bottom_corner_mid, mid_point_mid)

    corners_corrected = np.zeros_like(corners)
    for z in range(data.shape[-1]):
        if np.sum(left_contact[...,z])>10:
            data_shifted, trans_x, trans_y = shift_image(left_contact, z, trans_y=0, return_trans=True)
            top_corner, bottom_corner, mid_point = get_corners(data_shifted)

            angle, dist = get_measurements(top_corner, bottom_corner, mid_point)
            if (np.abs(angle - ref_angle) / ref_angle < 0.1 and np.abs(dist - ref_dist) / ref_dist < 0.15):
                corners_corrected[...,z] = corners[...,z] 

            data_shifted, trans_x, trans_y = shift_image(right_contact, z, trans_y=0, return_trans=True)
            top_corner, bottom_corner, mid_point = get_corners(data_shifted)

            angle, dist = get_measurements(top_corner, bottom_corner, mid_point)
            if (np.abs(angle - ref_angle_right) / ref_angle_right < 0.1 and np.abs(dist - ref_dist_right) / ref_dist_right < 0.5):
                corners_corrected[mid_point[0], mid_point[1], z] = 5
                

    fov_mask = (data>0).astype(np.uint8)
    
    return corners_corrected, fov_mask