import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree


def sample_points_along_straight_line(P0, sweep_dir, s_min, s_max, ds=0.5):
    
    sweep_dir = np.asarray(sweep_dir)
    P0 = np.asarray(P0)

    # Ensure ordering
    if s_min > s_max:
        s_min, s_max = s_max, s_min

    # walk backward from P0 to s_min
    s_vals_neg = [0.0]
    s = 0.0
    while s > s_min:
        s -= ds
        s_vals_neg.append(s)

    # walk forward from P0 to s_max
    s_vals_pos = []
    s = 0.0
    while s < s_max:
        s += ds
        s_vals_pos.append(s)

    s_all = np.array(s_vals_neg[::-1] + s_vals_pos)

    # convert to 3D
    points_3D = P0 + np.outer(s_all, sweep_dir)

    points_3D_before = P0 + np.outer(s_vals_neg[::-1], sweep_dir)
    points_3D_after  = P0 + np.outer(s_vals_pos, sweep_dir)

    return points_3D, points_3D_before, points_3D_after
    

def world_to_vox(affine, x_world):
    """World (mm) -> voxel (continuous)"""
    inv = np.linalg.inv(affine)
    x_h = np.append(np.asarray(x_world, dtype=float), 1.0)
    v = inv @ x_h
    return v[:3]

def vox_to_world(affine, v_vox):
    """Voxel (continuous) -> world (mm)"""
    v_h = np.append(np.asarray(v_vox, dtype=float), 1.0)
    x = affine @ v_h
    return x[:3]

def vox_to_world_many(affine, ijk):
    """
    Vectorized version of vox_to_world for an (N,3) array of voxel coords.
    Same transform, much faster than looping.
    """
    ijk = np.asarray(ijk, dtype=float)
    ones = np.ones((ijk.shape[0], 1), dtype=float)
    v_h = np.concatenate([ijk, ones], axis=1)        # (N,4)
    x_h = (affine @ v_h.T).T                         # (N,4)
    return x_h[:, :3]

def walk_on_surface(
    surface_world,
    x0_world,
    n_world,
    dx_mm=1.0,
    n_steps=200,
    k_candidates=100,
    min_forward_mm=1e-6, #Slightly greater than 0 to allow for not going not moving at 0 indefintenyl
    snap_start=True,
    max_no_forward=10,
    radius_mm=2,
    len_max_mm=100
):
    surface_world = np.asarray(surface_world, float)
    n = np.asarray(n_world, float)
    n /= np.linalg.norm(n)

    tree = cKDTree(surface_world)

    # Start: snap to nearest surface sample (recommended)
    x = np.asarray(x0_world, float)
    _, idx0 = tree.query(x, k=1)
    idx0 = int(idx0)
    if snap_start:
        x = surface_world[idx0]

    pts = [x.copy()]
    idxs = [idx0]
    visited = {idx0}
    
    no_forward_count = 0  

    for step in range(n_steps):
        y = x + dx_mm * n

        # local candidate set near the predicted point
        # dists, idx = tree.query(y, k=min(k_candidates, len(surface_world)))
        idx = tree.query_ball_point(y, r=radius_mm)
        idx = np.atleast_1d(idx).astype(int)

        r = radius_mm
        while len(idx) == 0:
            r += 1
            idx = tree.query_ball_point(y, r=r)
            idx = np.atleast_1d(idx).astype(int)

        cand = surface_world[idx]  # (k,3)
        # print(cand)
        dists = np.linalg.norm(cand - y, axis=1)
        
        steps = cand - x           # (k,3)
        norms = np.linalg.norm(steps, axis=1, keepdims=True)
        norms[norms == 0] = 1.0   # avoid division by zero
        steps = steps / norms
        scores = steps @ n         # dot product (k,)

        # Invalidate candidates that are already visited
        visited_mask = np.fromiter((i in visited for i in idx), dtype=bool, count=len(idx))
        scores[visited_mask] = -np.inf
        
        # If everything is invalid, stop
        max_score = np.max(scores)
        
        if not np.isfinite(max_score):
            break

        # stop after max_no_forward consecutive iterations with no forward progress
        if max_score < min_forward_mm:
            no_forward_count += 1
            if no_forward_count >= max_no_forward:
                break
        else:
            no_forward_count = 0
            
        tol = 1e-9
        best_candidates = np.where(scores >= max_score - tol)[0]

        # Among best score ties, choose the closest to y (smallest dists)
        d_best = dists[best_candidates]
        min_d = np.min(d_best)

        dist_tol = 1e-9
        closest_candidates = best_candidates[np.where(d_best <= min_d + dist_tol)[0]]


        # randomly select one (if there is equality useful)
        best = np.random.choice(closest_candidates)
        if not np.isfinite(scores[best]):
            break  # no valid move
        
        idx_next = int(idx[best])
        visited.add(idx_next)

        x = surface_world[idx_next]
        if np.linalg.norm(x - x0_world) >= len_max_mm:
            # print(f'distance overcome: {np.linalg.norm(x - x0_world)}')
            break
        pts.append(x.copy())
        idxs.append(idx_next)


    # remove trailing points that had no forward progress
    if no_forward_count > 0:
        pts = pts[:-no_forward_count]

    if len(pts) == 0:
        return np.empty((0, 3))
    else:
        return np.vstack(pts)


def slerp(v0, v1, t):
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    theta = np.arccos(dot)

    if theta < 1e-6:
        return v0

    sin_theta = np.sin(theta)
    return (
        np.sin((1 - t) * theta) / sin_theta * v0 +
        np.sin(t * theta) / sin_theta * v1
    )


def slice_pose_torch(
    vol_t: torch.Tensor,           # (1,1,D,H,W) on GPU
    inv_affine: torch.Tensor,       # (4,4) world→voxel
    point_world: torch.Tensor,      # (3,)
    dir_world: torch.Tensor,        # (3,)
    inplane_world: torch.Tensor,    # (3,)
    dx_mm: float,
    dy_mm: float,
    H_out: int,
    W_out: int,
    P_origin=None,
    fov_mask=None,
    pad_value: float = 0.0,
    align_corners: bool = True,
    return_world_coords: bool = False,
    mode='bilinear'
):
    """
    GPU-only slice resampling from a 3D volume using grid_sample.
    """

    device = vol_t.device
    dtype  = vol_t.dtype

    _, _, D, H, W = vol_t.shape

    # -------------------------
    # 2) Pixel grid
    # -------------------------
    ys = torch.arange(H_out, device=device, dtype=dtype)
    xs = torch.arange(W_out, device=device, dtype=dtype)

    if P_origin is not None:
        y0 = P_origin[1].to(dtype)
        x0 = P_origin[0].to(dtype)
    else:
        y0 = H_out / 2.0
        x0 = W_out / 4.0

    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    Y = Y.to(dtype)
    X = X.to(dtype)

    offset_y = (Y - y0) * dy_mm
    offset_x = (X - x0) * dx_mm

    # -------------------------
    # 3) World coordinates
    # -------------------------
    pts_world = (
        point_world[None, None, :]
        + offset_y[..., None] * inplane_world
        + offset_x[..., None] * dir_world
    )  # (H_out, W_out, 3)

    # -------------------------
    # 4) World → voxel (i,j,k)
    # -------------------------
    ones = torch.ones((*pts_world.shape[:-1], 1), device=device, dtype=dtype)
    pts_h = torch.cat([pts_world, ones], dim=-1)           # (...,4)

    ijk = torch.matmul(
        pts_h.view(-1, 4),
        inv_affine.T
    ).view(H_out, W_out, 4)[..., :3]
    # ijk = torch.einsum("...j,ij->...i", pts_h, inv_affine)[..., :3]

    X_idx = ijk[..., 0]
    Y_idx = ijk[..., 1]
    Z_idx = ijk[..., 2]

    # -------------------------
    # 5) Normalize for grid_sample
    # -------------------------
    if align_corners:
        x_norm = 2.0 * X_idx / (W - 1) - 1.0
        y_norm = 2.0 * Y_idx / (H - 1) - 1.0
        z_norm = 2.0 * Z_idx / (D - 1) - 1.0
    else:
        x_norm = (X_idx + 0.5) / W * 2 - 1
        y_norm = (Y_idx + 0.5) / H * 2 - 1
        z_norm = (Z_idx + 0.5) / D * 2 - 1

    grid = torch.stack((x_norm, y_norm, z_norm), dim=-1)
    # grid = grid.unsqueeze(0).unsqueeze(1)  # (1,1,H,W,3)
    grid = grid[None, None]

    # -------------------------
    # 6) Sample
    # -------------------------
    slice_t = F.grid_sample(
        vol_t,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )  # (1,1,1,H,W)

    slice_2d = slice_t[0, 0, 0] # (H,W)

    # -------------------------
    # 7) Padding value handling
    # -------------------------
    if pad_value != 0.0:
        oob = (
            (X_idx < 0) | (X_idx > W - 1) |
            (Y_idx < 0) | (Y_idx > H - 1) |
            (Z_idx < 0) | (Z_idx > D - 1)
        )
        slice_2d = slice_2d.masked_fill(oob, pad_value)

    if fov_mask is not None:
        slice_2d *= fov_mask

    if return_world_coords:
        return slice_2d, pts_world
    return slice_2d


def slice_pose_torch_batched(
    vol_t,
    inv_affine,
    Ps,
    d_vecs,
    i_vecs,
    dx_mm,
    dy_mm,
    H_out,
    W_out,
    Y, 
    X, 
    P_origin,
    fov_mask=None,
    align_corners=True,
    mode='bilinear'
):

    device = vol_t.device
    dtype = vol_t.dtype

    N = Ps.shape[0]
    _, _, D, H, W = vol_t.shape

    # -------------------
    # pixel grid
    # -------------------
    if (Y is None) or (X is None):
        ys = torch.arange(H_out, device=device, dtype=dtype)
        xs = torch.arange(W_out, device=device, dtype=dtype)

        Y, X = torch.meshgrid(ys, xs, indexing="ij")

    y0 = P_origin[1]
    x0 = P_origin[0]

    offset_y = (Y - y0) * dy_mm
    offset_x = (X - x0) * dx_mm

    # expand for batch
    offset_y = offset_y[None, ..., None] #(1,H,W,1)
    offset_x = offset_x[None, ..., None] #(1,H,W,1)

    Ps = Ps[:, None, None, :] #(N,1,1,3)
    d_vecs = d_vecs[:, None, None, :] #(N,1,1,3)
    i_vecs = i_vecs[:, None, None, :] #(N,1,1,3)

    # -------------------
    # world coords
    # -------------------
    pts_world = (
        Ps
        + offset_y * i_vecs
        + offset_x * d_vecs
    )  # (N,H,W,3)

    # -------------------
    # world → voxel
    # -------------------
    R = inv_affine[:3, :3]
    t = inv_affine[:3, 3]

    ijk = torch.matmul(
        pts_world.view(-1,3), #(N × H × W, 3)
        R.T
    ) + t

    ijk = ijk.view(N, H_out, W_out, 3)

    X_idx = ijk[...,0]
    Y_idx = ijk[...,1]
    Z_idx = ijk[...,2]

    # -------------------
    # normalize grid
    # -------------------
    if align_corners:
        x_norm = 2*X_idx/(W-1) - 1
        y_norm = 2*Y_idx/(H-1) - 1
        z_norm = 2*Z_idx/(D-1) - 1
    else:
        x_norm = (X_idx+0.5)/W*2 - 1
        y_norm = (Y_idx+0.5)/H*2 - 1
        z_norm = (Z_idx+0.5)/D*2 - 1

    grid = torch.stack((x_norm, y_norm, z_norm), dim=-1)

    grid = grid[:, None]  # (N,1,H,W,3)

    # -------------------
    # grid sample: interpolates N slices simultaneously, all batches reference the same vol_t (saves memeory)
    # -------------------
    vol_batch = vol_t.expand(N, -1, -1, -1, -1)

    slices = F.grid_sample(
        vol_batch,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners
    )

    slices = slices[:,0,0]  # (N,H,W)

    if fov_mask is not None:
        slices = slices * fov_mask

    return slices

def slice_pose_torch_shared_volume_batched(
    vol_t,
    inv_affine,
    Ps,
    d_vecs,
    i_vecs,
    dx_mm,
    dy_mm,
    H_out,
    W_out,
    Y=None,
    X=None,
    P_origin=None,
    fov_mask=None,
    align_corners=True,
    mode='bilinear'
):
    """
    Slice B*N planes from one shared 3D volume.

    Args:
        vol_t:       (1, C, D, H, W) or (C, D, H, W)
        inv_affine:  (4, 4)
        Ps:          (B, N, 3)
        d_vecs:      (B, N, 3)
        i_vecs:      (B, N, 3)
        dx_mm:       scalar
        dy_mm:       scalar
        H_out:       output slice height
        W_out:       output slice width
        Y, X:        optional meshgrid, each (H_out, W_out)
        P_origin:    (2,) or (B, 2)
        fov_mask:    optional, shape:
                        - (H_out, W_out)
                        - (B, H_out, W_out)
                        - (B, N, H_out, W_out)
        align_corners: bool

    Returns:
        slices:      (B, N, H_out, W_out) if C == 1
                     otherwise (B, N, C, H_out, W_out)
    """

    # -------------------
    # normalize volume shape
    # -------------------
    if vol_t.ndim == 4:
        vol_t = vol_t.unsqueeze(0)  # -> (1,C,D,H,W)

    if Ps.ndim == 2:
        Ps = Ps.unsqueeze(0)  # -> (1,D,3)
        d_vecs = d_vecs.unsqueeze(0)  # -> (1,D,3)
        i_vecs = i_vecs.unsqueeze(0)  # -> (1,D,3)

    assert vol_t.ndim == 5, f"vol_t must be (1,C,D,H,W) or (C,D,H,W), got {vol_t.shape}"
    assert vol_t.shape[0] == 1, f"Expected shared volume with batch=1, got {vol_t.shape}"

    device = vol_t.device
    dtype = vol_t.dtype

    _, C, D, H, W = vol_t.shape
    B, N, _ = Ps.shape

    # -------------------
    # pixel grid
    # -------------------
    if (Y is None) or (X is None):
        ys = torch.arange(H_out, device=device, dtype=dtype)
        xs = torch.arange(W_out, device=device, dtype=dtype)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")  # (H_out, W_out)

    # -------------------
    # origin handling
    # -------------------
    if P_origin is None:
        raise ValueError("P_origin must be provided")

    if P_origin.ndim == 1:
        # shared origin for all batch items
        x0 = P_origin[0]
        y0 = P_origin[1]

        offset_y = (Y - y0) * dy_mm      # (H_out, W_out)
        offset_x = (X - x0) * dx_mm      # (H_out, W_out)

        offset_y = offset_y[None, None, ..., None]  # (1,1,H,W,1)
        offset_x = offset_x[None, None, ..., None]  # (1,1,H,W,1)

    elif P_origin.ndim == 2:
        assert P_origin.shape[0] == B, f"P_origin batch {P_origin.shape[0]} != Ps batch {B}"

        x0 = P_origin[:, 0]  # (B,)
        y0 = P_origin[:, 1]  # (B,)

        offset_y = (Y[None, ...] - y0[:, None, None]) * dy_mm   # (B,H,W)
        offset_x = (X[None, ...] - x0[:, None, None]) * dx_mm   # (B,H,W)

        offset_y = offset_y[:, None, ..., None]  # (B,1,H,W,1)
        offset_x = offset_x[:, None, ..., None]  # (B,1,H,W,1)

    else:
        raise ValueError(f"P_origin must be shape (2,) or (B,2), got {P_origin.shape}")

    # -------------------
    # expand vectors
    # -------------------
    Ps = Ps[:, :, None, None, :]         # (B,N,1,1,3)
    d_vecs = d_vecs[:, :, None, None, :] # (B,N,1,1,3)
    i_vecs = i_vecs[:, :, None, None, :] # (B,N,1,1,3)

    # -------------------
    # world coords
    # -------------------
    pts_world = (
        Ps
        + offset_y * i_vecs
        + offset_x * d_vecs
    )  # (B,N,H_out,W_out,3)

    # -------------------
    # world -> voxel
    # -------------------
    R = inv_affine[:3, :3]   # (3,3)
    t = inv_affine[:3, 3]    # (3,)

    pts_flat = pts_world.reshape(-1, 3)           # (B*N*H_out*W_out, 3)
    ijk = torch.matmul(pts_flat, R.T) + t         # (B*N*H_out*W_out, 3)
    ijk = ijk.reshape(B, N, H_out, W_out, 3)      # (B,N,H_out,W_out,3)

    X_idx = ijk[..., 0]
    Y_idx = ijk[..., 1]
    Z_idx = ijk[..., 2]

    # -------------------
    # normalize grid
    # -------------------
    if align_corners:
        x_norm = 2 * X_idx / (W - 1) - 1
        y_norm = 2 * Y_idx / (H - 1) - 1
        z_norm = 2 * Z_idx / (D - 1) - 1
    else:
        x_norm = (X_idx + 0.5) / W * 2 - 1
        y_norm = (Y_idx + 0.5) / H * 2 - 1
        z_norm = (Z_idx + 0.5) / D * 2 - 1

    grid = torch.stack((x_norm, y_norm, z_norm), dim=-1)  # (B,N,H_out,W_out,3)
    grid = grid.reshape(B * N, 1, H_out, W_out, 3)        # (B*N,1,H_out,W_out,3)

    # -------------------
    # expand shared volume across all B*N slices
    # -------------------
    vol_batch = vol_t.expand(B * N, -1, -1, -1, -1)       # (B*N,C,D,H,W)

    # -------------------
    # sample
    # -------------------
    slices = F.grid_sample(
        vol_batch,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )  # (B*N,C,1,H_out,W_out)

    slices = slices.squeeze(2)  # (B*N,C,H_out,W_out)
    slices = slices.reshape(B, N, C, H_out, W_out)

    if C == 1:
        slices = slices[:, :, 0]  # (B,N,H_out,W_out)

    # -------------------
    # apply FOV mask
    # -------------------
    if fov_mask is not None:
        if fov_mask.ndim == 2:
            # (H,W) -> broadcast to (B,N,H,W)
            slices = slices * fov_mask

        elif fov_mask.ndim == 3:
            # assume (B,H,W) -> broadcast over N
            assert fov_mask.shape[0] == B, f"fov_mask batch {fov_mask.shape[0]} != B {B}"
            slices = slices * fov_mask[:, None]

        elif fov_mask.ndim == 4:
            # assume (B,N,H,W)
            assert fov_mask.shape[:2] == (B, N), f"fov_mask shape {fov_mask.shape[:2]} != {(B,N)}"
            slices = slices * fov_mask

        else:
            raise ValueError(f"Unsupported fov_mask shape: {fov_mask.shape}")

    return slices


