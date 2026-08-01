import os
import glob
import argparse
from dataclasses import dataclass, field
from itertools import chain, combinations
from typing import Dict, List, Tuple, Optional, Sequence
import pandas as pd
from tqdm import tqdm

import SimpleITK as sitk
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries

from utilities.generatesweep import *
from utilities.utils import set_determinism, save
from utilities.dataset import *
from networks.mhvae_fast import MHVAE2D


ALL_CASES = [
    "Case011", "Case025", "Case027", "Case045", "Case052", "Case056",
    "Case070", "Case074", "Case085", "Case099", "Case103", "Case112", "Case114"
]


@dataclass
class SyntheticSweepConfig:
    """
    Configuration object for synthetic MRI->US sweep generation.
    """
    path_data_mri: str = "./data/coregistered/mri-space"
    path_registration: str = "./data/registration/mni"
    path_data_us: str = "./data/coregistered/us-space"
    path_skull_strip: str = "./experiments/skullstripping_hdbet/"
    path_mni_surface: str = "./data/mni/mni_brain_surface.nii.gz.seg.nrrd"
    path_mni_non_viable: str = "./data/mni/refined_non_viable_surface_mask2.nii.gz"
    path_precomputed_us_masks: str = "./data/precomputed_us_masks"

    output_template: str = "./experiments/synthetic/{}-{}-{}/{}/{}/{}"

    dx_mm: float = 0.5
    dy_mm: float = 0.5
    out_h: int = 192
    out_w: int = 192

    skull_non_viable_dilation_iters: int = 5
    tumor_center_noise_std: float = 3.0
    angle_interp_fraction_max: float = 0.25
    angle_interp_num_steps: int = 100

    valid_cases: Tuple[str, ...] = tuple(ALL_CASES)


@dataclass
class SynthesisConfig:
    """
    Configuration for MRI -> synthetic US synthesis.
    """
    model_dir: str
    case: str

    type_normalization: str = "min-max"
    seed: int = 3
    temp_values: Sequence[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 1.0])

    max_features: int = 128
    base_features: int = 16
    nb_finalblocks: int = 6
    nfeat_finalblock: int = 8
    pools: int = 6
    epoch_inf: int = 1000
    workers: int = 10
    spatial_shape: Tuple[int, int] = (192, 192)
    modalities: Sequence[str] = field(default_factory=lambda: ["us", "t2", "cet1", "flair"])

    no_se: bool = True
    no_res: bool = True


@dataclass
class SweepCaseContext:
    """
    Precomputed case-specific state used for sweep generation.

    This object stores all geometry and MRI data needed to sample synthetic
    sweeps repeatedly for a single case.
    """
    case: str
    annotator: str
    modalities: List[str]
    total_subsets_mr: List[Tuple[str, ...]]
    mod_vols: Dict
    ultrasounds: List[str]
    filtered_surface_world: np.ndarray
    surface_weights: np.ndarray
    tumor_center_world: np.ndarray
    tumor_points_world: np.ndarray
    radius_mm: float


class SyntheticSweepGenerator:
    """
    Class-based pipeline for generating synthetic ultrasound sweeps from MRI volumes.

    Main responsibilities:
      1. Load MRI modalities and segmentation
      2. Compute valid probe entry points on the skull/brain surface
      3. Build synthetic sweep trajectories
      4. Resample MRI modalities and segmentation into US-like sweep space
      5. Save generated outputs to disk
    """

    def __init__(self, config: SyntheticSweepConfig):
        self.cfg = config

    def generate_case(
        self,
        case: str,
        annotator: str = "n1",
        num_sweeps_per_subset: int = 1,
        seed: int = 0,
    ) -> None:
        """
        Generate synthetic sweep data for one case.

        Parameters
        ----------
        case : str
            Case identifier, e.g. "Case027"
        annotator : str
            Segmentation annotator ID
        num_sweeps_per_subset : int
            Number of synthetic sweeps to generate per MRI modality subset
        seed : int
            Random seed for reproducibility
        """
        self._validate_case(case)
        set_determinism(seed=seed)

        print(f"[INFO] Generating data for case={case}, annotator={annotator}, K={num_sweeps_per_subset}, seed={seed}")

        # 1) Load case data
        modalities, modality_volumes = self._load_modalities_for_case(case)
        modality_subsets = self._get_all_modality_subsets(modalities)

        print(f"[INFO] Found {len(modality_subsets)} modality subsets from modalities={modalities}")

        _, reference_mri_sitk, _ = self._load_reference_mri(case)
        target_info = self._load_target_segmentation(case, annotator)

        # 2) Create output directories
        self._create_output_folders(annotator, num_sweeps_per_subset, seed, case)

        # 3) Get candidate ultrasound cases
        candidate_ultrasounds = self._get_candidate_ultrasounds(case)
        if len(candidate_ultrasounds) < num_sweeps_per_subset:
            raise ValueError(
                f"Requested {num_sweeps_per_subset} sweeps, but only {len(candidate_ultrasounds)} candidate US cases are available."
            )

        # 4) Compute viable probe entry points
        candidate_surface_world = self._compute_brain_surface_points_in_world(case, target_info["path"])
        filtered_surface_world, radius_mm = self._compute_filtered_viable_surface_points(
            case=case,
            img_mri=reference_mri_sitk,
            candidate_surface_world=candidate_surface_world
        )

        if len(filtered_surface_world) == 0:
            raise RuntimeError(f"No viable surface points found for case {case}")

        # 5) Tumor geometry
        tumor_voxels, tumor_points_world = self._compute_tumor_world_points(
            target_info["data"],
            target_info["nib"].affine
        )

        if len(tumor_points_world) == 0:
            raise RuntimeError(f"No tumor voxels found in segmentation for case {case}")

        tumor_center_world = self._compute_noisy_tumor_center_world(
            target_info["data"],
            target_info["nib"].affine
        )

        surface_weights = self._compute_surface_sampling_weights(
            filtered_surface_world,
            tumor_center_world
        )

        # 6) Generate sweeps for each subset of modalities
        for modalities_set in modality_subsets:
            print(f"[INFO] Generating for modality subset: {modalities_set}")

            selected_ultrasounds = iter(
                np.random.choice(candidate_ultrasounds, num_sweeps_per_subset, replace=False).tolist()
            )

            generated_count = 0
            while generated_count < num_sweeps_per_subset:
                us_case = next(selected_ultrasounds)

                # Sample viable entry point on surface
                chosen_idx = np.random.choice(filtered_surface_world.shape[0], p=surface_weights)
                entry_point_world = filtered_surface_world[chosen_idx]

                # Load US support data
                device = target_info["volume"].device
                keypoints, frames_with_surface, fov_mask = self._load_ultrasound_support_data(us_case, device)

                if len(frames_with_surface) == 0:
                    print(f"[WARN] No surface frames found for US case {us_case}, skipping")
                    continue

                _, p_us, fov_slice = self._choose_ultrasound_reference_frame(
                    keypoints,
                    frames_with_surface,
                    fov_mask,
                    device
                )

                # Construct sweep
                ps, d_vecs, i_vecs = self._build_full_sweep(
                    entry_point_world=entry_point_world,
                    tumor_center_world=tumor_center_world,
                    tumor_points_world=tumor_points_world,
                    filtered_surface_world=filtered_surface_world,
                    radius_mm=radius_mm
                )

                # Save MRI modalities
                self._save_modality_sweeps(
                    modalities_set=modalities_set,
                    modality_volumes=modality_volumes,
                    ps=ps,
                    d_vecs=d_vecs,
                    i_vecs=i_vecs,
                    p_us=p_us,
                    fov_slice=fov_slice,
                    case=case,
                    us_case=us_case,
                    annotator=annotator,
                    num_sweeps=num_sweeps_per_subset,
                    seed=seed
                )

                # Save segmentation
                self._save_segmentation_sweep(
                    target_info=target_info,
                    ps=ps,
                    d_vecs=d_vecs,
                    i_vecs=i_vecs,
                    p_us=p_us,
                    fov_slice=fov_slice,
                    case=case,
                    us_case=us_case,
                    modalities_set=modalities_set,
                    annotator=annotator,
                    num_sweeps=num_sweeps_per_subset,
                    seed=seed
                )

                generated_count += 1
                print(f"[INFO] Saved sweep {generated_count}/{num_sweeps_per_subset} for subset {modalities_set}")

        print("[INFO] Generation complete.")

    def synthesize_case(
        self,
        case: str,
        annotator: str,
        num_sweeps_per_subset: int,
        seed: int,
        synthesis_cfg: "SynthesisConfig",
    ) -> None:
        """
        Run the MRI -> US synthesis stage on already generated MRI sweep volumes.

        Inputs are read from:
            .../{case}/data_mri/

        Outputs are written to:
            .../{case}/data_us/
        """
        self._validate_case(case)
        set_determinism(seed=synthesis_cfg.seed)

        input_path = self._output_path(
            annotator, num_sweeps_per_subset, seed, case, "data_mri"
        )
        output_path = self._output_path(
            annotator, num_sweeps_per_subset, seed, case, "data_us"
        )

        os.makedirs(output_path, exist_ok=True)

        print(f"[INFO] Running synthesis for case={case}")
        print(f"[INFO] Synthesis input : {input_path}")
        print(f"[INFO] Synthesis output: {output_path}")

        self._check_case_not_seen_during_training(
            model_dir=synthesis_cfg.model_dir,
            case=synthesis_cfg.case
        )

        device = self._get_cuda_device()
        paths_dict = self._build_synthesis_paths_dict(
            input_path=input_path,
            modalities=synthesis_cfg.modalities
        )

        model = self._build_synthesis_model(device, synthesis_cfg)
        self._load_synthesis_weights(model, synthesis_cfg)

        self._run_synthesis_inference(
            paths_dict=paths_dict,
            saving_path=output_path,
            model=model,
            device=device,
            synthesis_cfg=synthesis_cfg
        )

        print("[INFO] Synthesis complete.")

    def generate_case_and_synthesize(
        self,
        case: str,
        annotator: str = "n1",
        num_sweeps_per_subset: int = 1,
        seed: int = 0,
        synthesis_cfg: Optional["SynthesisConfig"] = None,
    ) -> None:
        """
        Full pipeline:
        1. Generate MRI sweep volumes
        2. Run MRI -> US synthesis on generated sweeps
        """
        self.generate_case(
            case=case,
            annotator=annotator,
            num_sweeps_per_subset=num_sweeps_per_subset,
            seed=seed,
        )

        if synthesis_cfg is None:
            raise ValueError("synthesis_cfg must be provided for generate_case_and_synthesize().")

        self.synthesize_case(
            case=case,
            annotator=annotator,
            num_sweeps_per_subset=num_sweeps_per_subset,
            seed=seed,
            synthesis_cfg=synthesis_cfg,
        )

    # ----------------------------------------------------------------------------------
    # Path / folder helpers
    # ----------------------------------------------------------------------------------

    def _validate_case(self, case: str) -> None:
        if case not in self.cfg.valid_cases:
            raise ValueError(f"Case {case} is not in valid cases: {self.cfg.valid_cases}")

    def _output_path(
        self,
        annotator: str,
        num_sweeps: int,
        seed: int,
        case: str,
        folder: str,
        filename: str = ""
    ) -> str:
        return self.cfg.output_template.format(annotator, num_sweeps, seed, case, folder, filename)

    def _create_output_folders(self, annotator: str, num_sweeps: int, seed: int, case: str) -> None:
        for folder in ["data_mri", "data_us"]:
            os.makedirs(self._output_path(annotator, num_sweeps, seed, case, folder), exist_ok=True)

    # ----------------------------------------------------------------------------------
    # Synthesizer helpers
    # ----------------------------------------------------------------------------------

    def _get_cuda_device(self) -> torch.device:
        """
        Use CUDA for synthesis, matching the original script behavior.
        """
        if torch.cuda.is_available():
            print("[INFO] GPU available.")
            return torch.device("cuda:0")
        raise RuntimeError("[INFO] No GPU found. Synthesis requires CUDA in the current setup.")

    def _check_case_not_seen_during_training(self, model_dir: str, case: str) -> None:
        """
        Check that the synthesis model was not trained on the requested case.
        """
        split_path = os.path.join(model_dir, "split.csv")
        df_split = pd.read_csv(split_path, header=None)
        list_file_inference = df_split[df_split[1].isin(["inference"])][0].tolist()

        assert case + "-" in list_file_inference, (
            f"Synthesizer was trained with {case} - Forbidden"
        )
        print(f"[INFO] Check passed: synthesizer was not trained with {case}")

    def _build_synthesis_paths_dict(
        self,
        input_path: str,
        modalities: Sequence[str]
    ) -> List[Dict[str, str]]:
        """
        Build the list of case dictionaries expected by DatasetReMINDPred.

        Each dictionary maps modality name -> file path for one synthetic sweep sample.

        The function groups files by their common filename prefix before the final
        '_{modality}.nii.gz' suffix.
        """
        nii_files = [f for f in os.listdir(input_path) if f.endswith(".nii.gz")]

        groups = {}

        for filename in nii_files:
            matched_mod = None
            for mod in modalities:
                suffix = f"_{mod}.nii.gz"
                if filename.endswith(suffix):
                    matched_mod = mod
                    prefix = filename[: -len(suffix)]
                    break

            if matched_mod is None:
                # Ignore files such as *_seg.nii.gz
                continue

            if prefix not in groups:
                groups[prefix] = {}

            groups[prefix][matched_mod] = os.path.join(input_path, filename)

        paths_dict = list(groups.values())
        print(f"[INFO] Found {len(paths_dict)} synthetic MRI sweep samples for synthesis.")
        return paths_dict

    def _build_synthesis_model(
        self,
        device: torch.device,
        synthesis_cfg: "SynthesisConfig"
    ) -> torch.nn.Module:
        """
        Build the MHVAE2D synthesis model.
        """
        print("[INFO] Building hierarchical multi-modal synthesis model")

        model = MHVAE2D(
            modalities=synthesis_cfg.modalities,
            base_num_features=synthesis_cfg.base_features,
            num_pool=synthesis_cfg.pools,
            original_shape=synthesis_cfg.spatial_shape[:2],
            max_features=synthesis_cfg.max_features,
            with_residual=synthesis_cfg.no_res,
            with_se=synthesis_cfg.no_se,
            nb_finalblocks=synthesis_cfg.nb_finalblocks,
            nfeat_finalblock=synthesis_cfg.nfeat_finalblock,
        ).to(device)

        return model

    def _load_synthesis_weights(
        self,
        model: torch.nn.Module,
        synthesis_cfg: "SynthesisConfig"
    ) -> None:
        """
        Load pretrained synthesis model weights.
        """
        save_path = os.path.join(
            synthesis_cfg.model_dir,
            "models",
            f"CP_main_{synthesis_cfg.epoch_inf}.pth"
        )

        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Model weights not found: {save_path}")

        model.load_state_dict(torch.load(save_path))
        print(f"[INFO] Loaded synthesis model from {save_path}")

    # ----------------------------------------------------------------------------------
    # Data loading
    # ----------------------------------------------------------------------------------

    def _load_nifti_as_torch_volume(self, path: str):
        """
        Load a NIfTI image and convert it to torch volume with shape [1,1,Z,Y,X].
        """
        img_nib = nib.load(path)
        data = img_nib.get_fdata()

        vol_np = np.transpose(data, (2, 1, 0))
        vol_torch = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0)

        affine = torch.from_numpy(img_nib.affine)
        inv_affine = torch.linalg.inv(affine)

        return img_nib, data, vol_torch, affine, inv_affine

    def _load_reference_mri(self, case: str):
        path_mri = glob.glob(os.path.join(self.cfg.path_data_mri, case, f"{case}-**_ref.nii.gz"))[0]
        img_sitk = sitk.ReadImage(path_mri)
        img_nib = nib.load(path_mri)
        return path_mri, img_sitk, img_nib

    def _load_modalities_for_case(self, case: str):
        images_modalities = get_coregistered_mr_images(self.cfg.path_data_mri, case)
        modalities = [k[0] for k in images_modalities]

        modality_volumes = {}
        for mod in modalities:
            path_mod = glob.glob(os.path.join(self.cfg.path_data_mri, case, f"{case}-{mod}**.nii.gz"))[0]
            img_nib, _, vol_torch, _, inv_affine = self._load_nifti_as_torch_volume(path_mod)

            modality_volumes[mod] = {
                "volume": vol_torch,
                "affine": img_nib.affine,
                "inv_affine": inv_affine
            }

        return modalities, modality_volumes

    def _load_target_segmentation(self, case: str, annotator: str):
        path_target = os.path.join(self.cfg.path_data_mri, case, f"{case}-target_{annotator}.nii.gz")
        target_sitk = sitk.ReadImage(path_target)
        target_nib, target_data, target_vol_torch, target_affine, inv_target_affine = self._load_nifti_as_torch_volume(path_target)

        return {
            "path": path_target,
            "sitk": target_sitk,
            "nib": target_nib,
            "data": target_data,
            "volume": target_vol_torch,
            "affine": target_affine,
            "inv_affine": inv_target_affine
        }

    def _get_candidate_ultrasounds(self, case: str) -> List[str]:
        return [k for k in os.listdir(self.cfg.path_data_us) if "Case" in k and case not in k]

    def _get_all_modality_subsets(self, modalities: List[str]):
        return list(chain.from_iterable(
            combinations(modalities, r) for r in range(1, len(modalities) + 1)
        ))

    # ----------------------------------------------------------------------------------
    # Surface and tumor geometry
    # ----------------------------------------------------------------------------------

    def _compute_brain_surface_points_in_world(self, case: str, path_target: str):
        """
        Compute boundary voxels from the skull stripping mask and map them to world coordinates.
        These act as candidate probe entry points.
        """
        path_mask = glob.glob(os.path.join(self.cfg.path_skull_strip, case, f"{case}-**_mask.nii.gz"))[0]
        non_cerebrum_mask = (nib.load(path_mask).get_fdata() == 0)

        border = find_boundaries(non_cerebrum_mask, mode="inner")
        border_voxels = np.stack(np.where(border), axis=-1)

        target_affine = nib.load(path_target).affine
        border_world = vox_to_world_many(target_affine, border_voxels)

        return border_world

    def _compute_filtered_viable_surface_points(
        self,
        case: str,
        img_mri: sitk.Image,
        candidate_surface_world: np.ndarray
    ):
        """
        Remove surface points near non-viable craniotomy locations defined in MNI space.
        """
        affine_path = os.path.join(self.cfg.path_registration, case, f"{case}-mri-to-mni-Syn0GenericAffine.mat")
        sitk_affine = sitk.ReadTransform(affine_path)

        mni_surface_img = sitk.ReadImage(self.cfg.path_mni_surface)
        mni_surface_mask = sitk.GetArrayFromImage(mni_surface_img)[..., 0] > 0

        non_viable_mask = sitk.GetArrayFromImage(sitk.ReadImage(self.cfg.path_mni_non_viable)) > 0

        # kept for semantic clarity, though not directly used
        _viable_surface_mask = mni_surface_mask & (~non_viable_mask)

        non_viable_mask_dilated = binary_dilation(
            non_viable_mask,
            iterations=self.cfg.skull_non_viable_dilation_iters
        )
        non_viable_voxels = np.argwhere(non_viable_mask_dilated)

        non_viable_world = []
        for idx in non_viable_voxels:
            z, y, x = idx
            physical_point = mni_surface_img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
            transformed_point = sitk_affine.TransformPoint(physical_point)

            # Preserve original sign convention from source code
            non_viable_world.append([-1, -1, 1] * np.array(transformed_point))

        non_viable_world = np.array(non_viable_world)

        tree = cKDTree(non_viable_world)

        spacing = img_mri.GetSpacing()
        radius_mm = np.sqrt(spacing[0]**2 + spacing[1]**2 + spacing[2]**2)

        nearby_indices = tree.query_ball_point(candidate_surface_world, r=radius_mm)
        keep_mask = np.array([len(n) == 0 for n in nearby_indices])
        filtered_surface_world = candidate_surface_world[keep_mask]

        return filtered_surface_world, radius_mm

    def _compute_tumor_world_points(self, target_data: np.ndarray, target_affine: np.ndarray):
        tumor_voxels = np.argwhere(target_data > 0)
        tumor_world = vox_to_world_many(target_affine, tumor_voxels)
        return tumor_voxels, tumor_world

    def _compute_noisy_tumor_center_world(
        self,
        target_data: np.ndarray,
        target_affine: np.ndarray
    ):
        center_voxel = [coords.mean() for coords in np.where(target_data > 0)]
        center_voxel = np.array(center_voxel) + np.random.normal(
            loc=0.0,
            scale=self.cfg.tumor_center_noise_std,
            size=(3,)
        )
        center_world = target_affine.dot(center_voxel.tolist() + [1])[:3]
        return center_world

    def _compute_surface_sampling_weights(
        self,
        surface_points_world: np.ndarray,
        tumor_center_world: np.ndarray
    ):
        distances = np.linalg.norm(surface_points_world - tumor_center_world, axis=1)
        weights = np.exp(-distances)

        if weights.sum() == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= weights.sum()

        return weights

    # ----------------------------------------------------------------------------------
    # Ultrasound support data
    # ----------------------------------------------------------------------------------

    def _load_ultrasound_support_data(self, us_case: str, device: torch.device):
        keypoints_img = nib.load(
            os.path.join(self.cfg.path_precomputed_us_masks, us_case, f"{us_case}-keypoints.nii.gz")
        )
        keypoints = keypoints_img.get_fdata()

        frames_with_surface = np.argwhere(keypoints == 4)[:, 2]

        fov_img = nib.load(
            os.path.join(self.cfg.path_precomputed_us_masks, us_case, f"{us_case}-fov_mask.nii.gz")
        )
        fov_mask = torch.from_numpy(fov_img.get_fdata()).to(device)

        return keypoints, frames_with_surface, fov_mask

    def _choose_ultrasound_reference_frame(
        self,
        keypoints: np.ndarray,
        frames_with_surface: np.ndarray,
        fov_mask: torch.Tensor,
        device: torch.device
    ):
        """
        Choose a single US frame containing the skull/surface point and extract:
          - the 2D probe anchor point in image coordinates
          - the corresponding FOV mask slice
        """
        z_mid = np.random.choice(frames_with_surface)

        points_mask = keypoints[..., z_mid]
        fov_slice = fov_mask[..., z_mid]

        ys, xs = np.where(points_mask > 0)
        labels = points_mask[ys, xs]

        idx_surface = labels == 4
        p_us = np.array([xs[idx_surface][0], ys[idx_surface][0]])
        p_us = torch.from_numpy(p_us).to(device)

        return z_mid, p_us, fov_slice

    # ----------------------------------------------------------------------------------
    # Sweep construction
    # ----------------------------------------------------------------------------------

    def _compute_local_plane_basis(
        self,
        entry_point_world: np.ndarray,
        tumor_center_world: np.ndarray
    ):
        """
        Build the initial probe direction and an orthonormal plane basis around it.
        """
        d_vec_0 = tumor_center_world - entry_point_world
        d_vec_0 = d_vec_0 / np.linalg.norm(d_vec_0)

        n = d_vec_0
        min_idx = np.argmin(np.abs(n))
        v = np.zeros(3)
        v[min_idx] = 1.0

        e1 = np.cross(n, v)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n, e1)

        return d_vec_0, e1, e2

    def _compute_sweep_direction(
        self,
        entry_point_world: np.ndarray,
        tumor_points_world: np.ndarray,
        tumor_center_world: np.ndarray
    ):
        """
        Estimate a plausible sweep direction lying on the local tangent plane.
        The direction is sampled based on PCA of tumor spread projected onto that plane.
        """
        d_vec_0, e1, e2 = self._compute_local_plane_basis(entry_point_world, tumor_center_world)

        tumor_points_2d = []
        for pt in tumor_points_world:
            rel = pt - entry_point_world
            x_prime = np.dot(rel, e1)
            y_prime = np.dot(rel, e2)
            tumor_points_2d.append([x_prime, y_prime])

        tumor_points_2d = np.array(tumor_points_2d)

        pca = PCA(n_components=2)
        pca.fit(tumor_points_2d)

        variance = pca.explained_variance_
        components = pca.components_

        pc1_3d = (components[0, 0] * e1) + (components[0, 1] * e2)
        pc1_3d /= np.linalg.norm(pc1_3d)

        pc2_3d = (components[1, 0] * e1) + (components[1, 1] * e2)
        pc2_3d /= np.linalg.norm(pc2_3d)

        x = np.random.normal(0, np.sqrt(variance[0]))
        y = np.random.normal(0, np.sqrt(variance[1]))

        sweep_dir = (x * pc1_3d) + (y * pc2_3d)
        sweep_dir /= np.linalg.norm(sweep_dir)

        return d_vec_0, sweep_dir

    def _compute_tumor_extent_along_direction(
        self,
        entry_point_world: np.ndarray,
        tumor_points_world: np.ndarray,
        sweep_dir: np.ndarray
    ):
        tumor_rel = tumor_points_world - entry_point_world
        tumor_proj_1d = np.dot(tumor_rel, sweep_dir)

        tumor_min = np.min(tumor_proj_1d)
        tumor_max = np.max(tumor_proj_1d)

        return tumor_min, tumor_max, tumor_proj_1d

    def _build_probe_orientation_sequence(
        self,
        entry_point_world: np.ndarray,
        d_vec_0: np.ndarray,
        sweep_dir: np.ndarray
    ):
        """
        Create the initial orientation-only part of the sweep.
        Probe position is fixed while direction is tilted.
        """
        i_vec_0 = np.cross(d_vec_0, sweep_dir)
        i_vec_0 = i_vec_0 / np.linalg.norm(i_vec_0)

        ps = [entry_point_world]
        d_vecs = []
        i_vecs = []

        num_steps = self.cfg.angle_interp_num_steps
        max_fraction = self.cfg.angle_interp_fraction_max

        # Tilt toward +sweep_dir
        for i in range(num_steps):
            t = i / (num_steps - 1)
            if t > max_fraction:
                break

            d_vec = slerp(d_vec_0, sweep_dir, t)
            d_vec /= np.linalg.norm(d_vec)

            d_vecs.append(d_vec)
            i_vecs.append(i_vec_0)

        ps = np.vstack([ps, np.repeat(ps[-1][None, :], i - 1, axis=0)])

        # Return from tilt
        d_vecs += d_vecs[::-1]
        i_vecs += i_vecs[::-1]
        ps = np.vstack([ps, np.repeat(ps[-1][None, :], i, axis=0)])

        # Tilt toward -sweep_dir
        d_vecs_init = []
        i_vecs_init = []

        for i in range(num_steps):
            t = i / (num_steps - 1)
            if t > max_fraction:
                break

            d_vec = slerp(d_vec_0, -sweep_dir, t)
            d_vec /= np.linalg.norm(d_vec)

            d_vecs_init.append(d_vec)
            i_vecs_init.append(i_vec_0)

        ps = np.vstack([ps, np.repeat(ps[-1][None, :], i, axis=0)])

        d_vecs_init += d_vecs_init[::-1]
        i_vecs_init += i_vecs_init[::-1]

        ps = np.vstack([ps, np.repeat(ps[-1][None, :], i, axis=0)])

        d_vecs = d_vecs_init + d_vecs
        i_vecs = i_vecs_init + i_vecs

        return ps, d_vecs, i_vecs, i_vec_0

    def _extend_sweep_along_surface(
        self,
        entry_point_world: np.ndarray,
        sweep_dir: np.ndarray,
        tumor_min: float,
        tumor_max: float,
        filtered_surface_world: np.ndarray,
        radius_mm: float,
        d_vec_0: np.ndarray,
        i_vec_0: np.ndarray,
        ps,
        d_vecs,
        i_vecs
    ):
        """
        Move the probe along the surface while keeping the plane orientation constant.
        """
        _, ps_before, ps_after = sample_points_along_straight_line(
            entry_point_world, sweep_dir, tumor_min, tumor_max
        )

        ps_after = walk_on_surface(
            filtered_surface_world,
            x0_world=entry_point_world,
            n_world=sweep_dir,
            dx_mm=self.cfg.dx_mm,
            n_steps=len(ps_after),
            len_max_mm=np.abs(tumor_max),
            radius_mm=radius_mm,
        )

        ps_before = walk_on_surface(
            filtered_surface_world,
            x0_world=entry_point_world,
            n_world=-sweep_dir,
            dx_mm=self.cfg.dx_mm,
            n_steps=len(ps_before),
            len_max_mm=np.abs(tumor_min),
            radius_mm=radius_mm,
        )

        ps_before = ps_before[::-1][1:]

        d_vecs_before = [d_vec_0 for _ in range(ps_before.shape[0])]
        i_vecs_before = [i_vec_0 for _ in range(ps_before.shape[0])]
        d_vecs_after = [d_vec_0 for _ in range(ps_after.shape[0])]
        i_vecs_after = [i_vec_0 for _ in range(ps_after.shape[0])]

        ps = np.vstack([ps_before, ps, ps_after])
        d_vecs = d_vecs_before + d_vecs + d_vecs_after
        i_vecs = i_vecs_before + i_vecs + i_vecs_after

        return ps, d_vecs, i_vecs

    def _append_terminal_angle_sequences(
        self,
        ps,
        d_vecs,
        i_vecs,
        d_vec_0: np.ndarray,
        sweep_dir: np.ndarray,
        i_vec_0: np.ndarray
    ):
        """
        Add orientation changes at both ends of the final sweep.
        """
        num_steps = self.cfg.angle_interp_num_steps
        max_fraction = self.cfg.angle_interp_fraction_max

        # End tilt toward +sweep_dir
        for i in range(num_steps):
            t = i / (num_steps - 1)
            if t > max_fraction:
                break

            d_vec = slerp(d_vec_0, sweep_dir, t)
            d_vec /= np.linalg.norm(d_vec)

            d_vecs.append(d_vec)
            i_vecs.append(i_vec_0)

        ps = np.vstack([ps, np.repeat(ps[-1][None, :], i, axis=0)])

        # Prepend reversed tilt from -sweep_dir
        d_vecs_init = []
        i_vecs_init = []

        for i in range(num_steps):
            t = i / (num_steps - 1)
            if t > max_fraction:
                break

            d_vec = slerp(d_vec_0, -sweep_dir, t)
            d_vec /= np.linalg.norm(d_vec)

            d_vecs_init.append(d_vec)
            i_vecs_init.append(i_vec_0)

        d_vecs_init.reverse()
        i_vecs_init.reverse()

        d_vecs = d_vecs_init + d_vecs
        i_vecs = i_vecs_init + i_vecs

        ps = np.vstack([np.repeat(ps[0][None, :], i, axis=0), ps])

        return ps, d_vecs, i_vecs

    def _build_full_sweep(
        self,
        entry_point_world: np.ndarray,
        tumor_center_world: np.ndarray,
        tumor_points_world: np.ndarray,
        filtered_surface_world: np.ndarray,
        radius_mm: float
    ):
        """
        Build the full sweep trajectory:
          - pick a sweep direction
          - create initial orientation changes
          - translate probe along surface
          - append terminal orientation changes
        """
        d_vec_0, sweep_dir = self._compute_sweep_direction(
            entry_point_world,
            tumor_points_world,
            tumor_center_world
        )

        tumor_min, tumor_max, _ = self._compute_tumor_extent_along_direction(
            entry_point_world,
            tumor_points_world,
            sweep_dir
        )

        ps, d_vecs, i_vecs, i_vec_0 = self._build_probe_orientation_sequence(
            entry_point_world,
            d_vec_0,
            sweep_dir
        )

        ps, d_vecs, i_vecs = self._extend_sweep_along_surface(
            entry_point_world=entry_point_world,
            sweep_dir=sweep_dir,
            tumor_min=tumor_min,
            tumor_max=tumor_max,
            filtered_surface_world=filtered_surface_world,
            radius_mm=radius_mm,
            d_vec_0=d_vec_0,
            i_vec_0=i_vec_0,
            ps=ps,
            d_vecs=d_vecs,
            i_vecs=i_vecs
        )

        ps, d_vecs, i_vecs = self._append_terminal_angle_sequences(
            ps, d_vecs, i_vecs, d_vec_0, sweep_dir, i_vec_0
        )

        return ps, d_vecs, i_vecs

    # ----------------------------------------------------------------------------------
    # Resampling / saving
    # ----------------------------------------------------------------------------------

    def _resample_volume_along_sweep(
        self,
        volume_torch: torch.Tensor,
        inv_affine: torch.Tensor,
        ps_torch: torch.Tensor,
        d_vecs_torch: torch.Tensor,
        i_vecs_torch: torch.Tensor,
        p_us: torch.Tensor,
        fov_slice: torch.Tensor,
        device: torch.device,
        interpolation_mode: str
    ):
        """
        Resample a 3D MRI/segmentation volume into a stack of 2D slices following the sweep.
        """
        slices = []

        for idx in range(ps_torch.shape[0]):
            point_world = ps_torch[idx].to(device)
            dir_world = d_vecs_torch[idx].to(device)
            inplane_world = i_vecs_torch[idx].to(device)

            slice_2d_torch, _ = slice_pose_torch(
                volume_torch,
                inv_affine,
                point_world=point_world,
                dir_world=dir_world,
                inplane_world=inplane_world,
                dx_mm=self.cfg.dx_mm,
                dy_mm=self.cfg.dy_mm,
                H_out=self.cfg.out_h,
                W_out=self.cfg.out_w,
                P_origin=p_us,
                fov_mask=fov_slice,
                pad_value=0.0,
                return_world_coords=True,
                mode=interpolation_mode
            )

            slices.append(slice_2d_torch.detach().cpu().numpy())

        return np.stack(slices, axis=-1)

    def _save_modality_sweeps(
        self,
        modalities_set,
        modality_volumes,
        ps,
        d_vecs,
        i_vecs,
        p_us,
        fov_slice,
        case: str,
        us_case: str,
        annotator: str,
        num_sweeps: int,
        seed: int
    ):
        """
        Resample and save each requested MRI modality for the synthetic sweep.
        """
        ps_torch = torch.as_tensor(ps)
        d_vecs_torch = torch.as_tensor(np.array(d_vecs))
        i_vecs_torch = torch.as_tensor(np.array(i_vecs))

        subset_name = "-".join([str(m) for m in modalities_set])
        filename_root = f"{case}-{us_case}-{subset_name}"

        for mod in modalities_set:
            vol_torch = modality_volumes[mod]["volume"]
            device = vol_torch.device
            inv_affine = modality_volumes[mod]["inv_affine"].to(device)

            volume_3d = self._resample_volume_along_sweep(
                volume_torch=vol_torch.to(device),
                inv_affine=inv_affine,
                ps_torch=ps_torch,
                d_vecs_torch=d_vecs_torch,
                i_vecs_torch=i_vecs_torch,
                p_us=p_us,
                fov_slice=fov_slice,
                device=device,
                interpolation_mode="bilinear"
            )

            output_path = self._output_path(
                annotator, num_sweeps, seed, case, "data_mri",
                filename_root + f"_{mod}.nii.gz"
            )

            nib.save(
                nib.Nifti1Image(volume_3d, modality_volumes[mod]["affine"]),
                output_path
            )

    def _save_segmentation_sweep(
        self,
        target_info,
        ps,
        d_vecs,
        i_vecs,
        p_us,
        fov_slice,
        case: str,
        us_case: str,
        modalities_set,
        annotator: str,
        num_sweeps: int,
        seed: int
    ):
        """
        Resample and save the target segmentation for the synthetic sweep.
        """
        ps_torch = torch.as_tensor(ps)
        d_vecs_torch = torch.as_tensor(np.array(d_vecs))
        i_vecs_torch = torch.as_tensor(np.array(i_vecs))

        vol_target = target_info["volume"]
        device = vol_target.device
        inv_affine_target = target_info["inv_affine"].to(device)

        volume_3d = self._resample_volume_along_sweep(
            volume_torch=vol_target,
            inv_affine=inv_affine_target,
            ps_torch=ps_torch,
            d_vecs_torch=d_vecs_torch,
            i_vecs_torch=i_vecs_torch,
            p_us=p_us,
            fov_slice=fov_slice,
            device=device,
            interpolation_mode="nearest"
        ).astype(np.uint8)

        subset_name = "-".join([str(m) for m in modalities_set])
        filename_root = f"{case}-{us_case}-{subset_name}"

        output_path = self._output_path(
            annotator, num_sweeps, seed, case, "data_mri",
            filename_root + "_seg.nii.gz"
        )

        nib.save(
            nib.Nifti1Image(volume_3d, target_info["nib"].affine),
            output_path
        )

    # ----------------------------------------------------------------------------------
    # Synthesis
    # ----------------------------------------------------------------------------------

    def _run_synthesis_inference(
        self,
        paths_dict,
        saving_path: str,
        model: torch.nn.Module,
        device: torch.device,
        synthesis_cfg: "SynthesisConfig"
    ) -> None:
        """
        Run MRI -> US synthesis inference over all generated MRI sweep samples.
        """
        print(f"[INFO] Synthesis modalities are {synthesis_cfg.modalities}")

        subjects_dataset = DatasetReMINDPred(
            paths_unnorm=paths_dict,
            normalization=True,
            type_normalization=synthesis_cfg.type_normalization,
        )
        dataloader = DataLoader(
            subjects_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=synthesis_cfg.workers
        )

        model.eval()

        for batch in tqdm(dataloader):
            imgs = {}
            nonempty_modalities = []

            for mod in synthesis_cfg.modalities:
                if mod in batch.keys():
                    imgs[mod] = batch[mod].to(device).permute(0, 4, 1, 2, 3)
                    imgs[mod] = imgs[mod].reshape(-1, 1, *batch[mod].shape[2:4])
                    nonempty_modalities.append(mod)

            if len(nonempty_modalities) == 0:
                continue

            first_mod = nonempty_modalities[0]

            # Use only MR modalities as encoder input
            subset_mr = [m for m in nonempty_modalities if m != "us"]
            if len(subset_mr) == 0:
                continue

            target_modality = model.modalities[0]  # expected to be 'us'

            with torch.inference_mode():
                affine = batch[f"{first_mod}_affine"][0].cpu().numpy().squeeze()
                name = batch[f"{first_mod}_name"][0].replace(f"_{first_mod}", "") + "_{}.nii.gz"

                model_input = {mod: imgs[mod].clone() for mod in subset_mr}

                encoded = model.encode(model_input)

                for temp in synthesis_cfg.temp_values:
                    pred, _, _ = model.decode(
                        encoded,
                        temp,
                        return_feat=True,
                        return_cat=True,
                        target_modality=target_modality,
                        compute_kl=False
                    )
                    save(pred, affine, os.path.join(saving_path, name.format(temp)))

    # ----------------------------------------------------------------------------------
    # Generate training data on-the-fly
    # ----------------------------------------------------------------------------------

    def prepare_case_context(
        self,
        case: str,
        annotator: str = "n1",
        drop_modalities: Optional[Sequence[str]] = None,
        target_name: str = "target",
    ) -> SweepCaseContext:
        """
        Prepare all reusable case-specific data required for repeated sweep sampling.

        This is useful for:
        - fixed validation set generation
        - on-the-fly training sample generation

        Parameters
        ----------
        case : str
            Case identifier.
        annotator : str
            Annotator used for target segmentation if needed.
        drop_modalities : Optional[Sequence[str]]
            Modalities to exclude from MRI input subsets.
        target_name : str
            Name of the target segmentation volume on disk.

        Returns
        -------
        SweepCaseContext
            Precomputed context for repeated sweep sampling.
        """
        self._validate_case(case)

        drop_modalities = set(drop_modalities or [])

        # Load available MRI modalities
        images_modalities = get_coregistered_mr_images(self.cfg.path_data_mri, case)
        modalities = [k[0] for k in images_modalities if k[0] not in drop_modalities]

        total_subsets_mr = self._get_all_modality_subsets(modalities)

        # Load MRI modality volumes
        mod_vols = {}
        for mod in modalities + [target_name]:
            path = glob.glob(os.path.join(self.cfg.path_data_mri, case, f"{case}-{mod}**.nii.gz"))[0]
            img_nib, _, vol_torch, _, inv_affine = self._load_nifti_as_torch_volume(path)
            mod_vols[mod] = {
                "volume": vol_torch.to(torch.float32),
                "affine": img_nib.affine,
                "inv_affine": inv_affine.to(torch.float32),
            }

        # Load reference MRI / target segmentation
        _, reference_mri_sitk, reference_mri_nib = self._load_reference_mri(case)
        target_path = os.path.join(self.cfg.path_data_mri, case, f"{case}-{target_name}_{annotator}.nii.gz")
        target_nib = nib.load(target_path)
        target_data = target_nib.get_fdata()

        # Candidate surface points
        candidate_surface_world = self._compute_brain_surface_points_in_world(case, target_path)
        filtered_surface_world, radius_mm = self._compute_filtered_viable_surface_points(
            case=case,
            img_mri=reference_mri_sitk,
            candidate_surface_world=candidate_surface_world
        )

        if len(filtered_surface_world) == 0:
            raise RuntimeError(f"No viable surface points found for case {case}")

        # Tumor geometry
        tumor_voxels = np.argwhere(target_data > 0)
        tumor_points_world = vox_to_world_many(target_nib.affine, tumor_voxels)

        if len(tumor_points_world) == 0:
            raise RuntimeError(f"No tumor voxels found for case {case}")

        tumor_center_vox = np.array([coords.mean() for coords in np.where(target_data > 0)])
        tumor_center_world = vox_to_world(target_nib.affine, tumor_center_vox)

        tumor_center_noisy_world = self._compute_noisy_tumor_center_world(
            target_data, target_nib.affine
        )

        surface_weights = self._compute_surface_sampling_weights(
            filtered_surface_world,
            tumor_center_noisy_world
        )

        ultrasounds = self._get_candidate_ultrasounds(case)

        return SweepCaseContext(
            case=case,
            annotator=annotator,
            modalities=modalities,
            total_subsets_mr=total_subsets_mr,
            mod_vols=mod_vols,
            ultrasounds=ultrasounds,
            filtered_surface_world=filtered_surface_world,
            surface_weights=surface_weights,
            tumor_center_world=tumor_center_world,
            tumor_points_world=tumor_points_world,
            radius_mm=radius_mm,
        )

    def sample_sweep(
        self,
        context: SweepCaseContext,
        us_case: str,
        device: torch.device,
    ):
        """
        Sample a single synthetic sweep trajectory and return the pose sequence and
        ultrasound reference slice info, without resampling MRI volumes yet.

        Returns
        -------
        ps_torch, d_vecs_torch, i_vecs_torch, p_us, fov_slice
        """
        chosen_idx = np.random.choice(
            context.filtered_surface_world.shape[0],
            p=context.surface_weights
        )
        entry_point_world = context.filtered_surface_world[chosen_idx]

        keypoints, frames_with_surface, fov_mask = self._load_ultrasound_support_data(
            us_case, device=torch.device("cpu")
        )

        if len(frames_with_surface) == 0:
            raise RuntimeError(f"No surface frames found for US case {us_case}")

        _, p_us, fov_slice = self._choose_ultrasound_reference_frame(
            keypoints,
            frames_with_surface,
            fov_mask,
            device=torch.device("cpu")
        )

        ps, d_vecs, i_vecs = self._build_full_sweep(
            entry_point_world=entry_point_world,
            tumor_center_world=context.tumor_center_world,
            tumor_points_world=context.tumor_points_world,
            filtered_surface_world=context.filtered_surface_world,
            radius_mm=context.radius_mm
        )

        return (
            torch.as_tensor(ps, dtype=torch.float32),
            torch.as_tensor(np.array(d_vecs), dtype=torch.float32),
            torch.as_tensor(np.array(i_vecs), dtype=torch.float32),
            p_us.to(torch.float32),
            fov_slice.to(torch.float32),
        )

    def render_sweep_volumes(
        self,
        context: SweepCaseContext,
        modalities_set: Sequence[str],
        ps: torch.Tensor,
        d_vecs: torch.Tensor,
        i_vecs: torch.Tensor,
        p_us: torch.Tensor,
        fov_slice: torch.Tensor,
        device: torch.device,
        include_target: bool = True,
        y_grid: Optional[torch.Tensor] = None,
        x_grid: Optional[torch.Tensor] = None,
    ):
        """
        Resample selected MRI modalities (and optionally target segmentation) along
        a sweep trajectory and return them in memory.

        Returns
        -------
        output : dict
            Keys are modality names; values are tensors shaped like the previous code.
        """
        output = {}

        mods_to_render = list(modalities_set)
        if include_target:
            mods_to_render.append("target")

        for mod in mods_to_render:
            vol_t = context.mod_vols[mod]["volume"].to(device)
            inv_affine = context.mod_vols[mod]["inv_affine"].to(device)
            interp_mode = "nearest" if mod == "target" else "bilinear"

            slices = slice_pose_torch_shared_volume_batched(
                vol_t,
                inv_affine,
                ps.to(device),
                d_vecs.to(device),
                i_vecs.to(device),
                dx_mm=self.cfg.dx_mm,
                dy_mm=self.cfg.dy_mm,
                H_out=self.cfg.out_h,
                W_out=self.cfg.out_w,
                Y=y_grid,
                X=x_grid,
                P_origin=p_us.to(device),
                fov_mask=fov_slice.to(device),
                mode=interp_mode,
            )

            output[mod] = slices.permute(0, 2, 3, 1)

        return output

    def generate_training_sample(
        self,
        context: SweepCaseContext,
        us_case: str,
        device: torch.device,
        y_grid: Optional[torch.Tensor] = None,
        x_grid: Optional[torch.Tensor] = None,
    ):
        """
        Generate one synthetic sweep sample in memory for training.

        Returns
        -------
        output : dict
            Resampled MRI modalities + target segmentation
        modalities_set : tuple[str, ...]
            Chosen MRI subset
        """
        modalities_set = random.choice(context.total_subsets_mr)

        ps, d_vecs, i_vecs, p_us, fov_slice = self.sample_sweep(
            context=context,
            us_case=us_case,
            device=device
        )

        output = self.render_sweep_volumes(
            context=context,
            modalities_set=modalities_set,
            ps=ps,
            d_vecs=d_vecs,
            i_vecs=i_vecs,
            p_us=p_us,
            fov_slice=fov_slice,
            device=device,
            include_target=True,
            y_grid=y_grid,
            x_grid=x_grid,
        )

        return output, modalities_set

    def _preprocess_mr_torch(
        self,
        x: torch.Tensor,
        k=(0, 0, 0),
        norm: bool = True,
        type_normalization: str = "standardization",
        mode: str = "training",
    ):
        """
        Torch preprocessing used before MRI->US synthesis.
        """
        LOWER = 0.0
        UPPER = 99.95

        if mode == "training":
            if k[0] == 1:
                x = torch.flip(x, [0])
            if k[1] == 1:
                x = torch.flip(x, [1])
            if k[2] == 1:
                x = torch.flip(x, [2])
            x = x.clone()

        mask = x > 0
        x = x.float()

        if norm:
            if torch.any(mask):
                if type_normalization == "standardization":
                    max_data = torch.quantile(x, UPPER / 100)
                    x[x > max_data] = max_data
                    sub = torch.mean(x)
                    div = 3 * torch.std(x)
                    x = (x - sub) / (div + 1e-8)
                    x[x > 1] = 1
                    x[~mask] = -1
                elif type_normalization == "min-max":
                    min_data = torch.quantile(x[mask], LOWER)
                    max_data = torch.quantile(x[mask], UPPER / 100)
                    x[x > max_data] = max_data
                    x = (x - min_data) / (max_data - min_data)
                    x = x * (1 + 255 / 256) - 255 / 256
                    x[~mask] = -1.0
                    div = (max_data - min_data) / 2
                    sub = min_data
                else:
                    raise NotImplementedError(type_normalization)
            else:
                x -= 1
                div = 1
                sub = 1
        else:
            if torch.any(mask):
                x = 2 * x - 1
                div = 0.5
                sub = 0.5
            else:
                x -= 1
                div = 1
                sub = 1

        return x, sub, div

    def synthesize_us_sweep_in_memory(
        self,
        synthesizer: torch.nn.Module,
        modalities,
        output,
        device: torch.device,
        type_normalization: str,
        saving_path: str = "",
        temperatures=(0.3, 0.5, 0.7, 1.0),
        mode: str = "training",
    ):
        """
        Synthesize US sweep from in-memory MRI sweep volumes.

        If mode != 'training' and saving_path is provided, save all temperatures.
        If mode == 'training', sample one temperature and return the prediction.
        """
        synthesizer.eval()

        imgs = {}
        nonempty_list = []

        for mod in modalities:
            img, _, _ = self._preprocess_mr_torch(
                output[mod].squeeze(),
                norm=True,
                type_normalization=type_normalization,
                mode=mode
            )
            output[mod] = img.unsqueeze(0).unsqueeze(0)
            imgs[mod] = output[mod].permute(0, 4, 1, 2, 3)
            imgs[mod] = imgs[mod].reshape(-1, 1, *output[mod].shape[2:4])
            nonempty_list.append(mod)

        subset_mr = [m for m in nonempty_list if m != "us"]

        target_modality = synthesizer.modalities[0]  # expected to be 'us'

        with torch.inference_mode():
            model_input = {mod: imgs[mod].clone() for mod in subset_mr}

            encoded = synthesizer.encode(model_input)

            temp = np.random.choice(list(temperatures))
            pred, _, _ = synthesizer.decode(
                encoded,
                temp,
                return_feat=True,
                return_cat=True,
                target_modality=target_modality,
                compute_kl=False
            )

            if mode == "training":
                return pred
            else:
                affine = np.eye(4)
                save(pred, affine, saving_path.format(temp))
                return None

    # ----------------------------------------------------------------------------------
    # Generate fixed validation dataset
    # ----------------------------------------------------------------------------------

    def generate_fixed_validation_dataset(
        self,
        context: SweepCaseContext,
        output_path: str,
        synthesizer: torch.nn.Module,
        num_samples: int,
        device: torch.device,
        type_normalization: str,
        temperatures=(0.3, 0.5, 0.7, 1.0),
    ):
        """
        Generate and store a fixed synthetic validation dataset.

        For each sample:
        1. sample a sweep
        2. render MRI sweep volumes to disk
        3. synthesize US sweep volumes to disk
        """
        os.makedirs(output_path, exist_ok=True)

        available_us = context.ultrasounds.copy()
        selected_us = iter(np.random.choice(available_us, num_samples + 10, replace=False).tolist())

        count = 0
        while count < num_samples:
            us_case = next(selected_us)

            modalities_set = random.choice(context.total_subsets_mr)

            ps, d_vecs, i_vecs, p_us, fov_slice = self.sample_sweep(
                context=context,
                us_case=us_case,
                device=device
            )

            output = self.render_sweep_volumes(
                context=context,
                modalities_set=modalities_set,
                ps=ps,
                d_vecs=d_vecs,
                i_vecs=i_vecs,
                p_us=p_us,
                fov_slice=fov_slice,
                device=device,
                include_target=True,
            )

            subset_name = "-".join(modalities_set)
            filename_root = f"{context.case}-{us_case}-{subset_name}"

            # Save MRI and segmentation
            for mod in list(modalities_set) + ["target"]:
                arr = output[mod].squeeze(0).detach().cpu().numpy()
                nib.save(
                    nib.Nifti1Image(arr, context.mod_vols[mod]["affine"]),
                    os.path.join(output_path, f"{filename_root}_{mod}.nii.gz")
                )

            # Synthesize and save US
            self.synthesize_us_sweep_in_memory(
                synthesizer=synthesizer,
                modalities=modalities_set,
                output=output,
                device=device,
                type_normalization=type_normalization,
                saving_path=os.path.join(output_path, f"{filename_root}" + "_{}.nii.gz"),
                temperatures=temperatures,
                mode="validation",
            )

            available_us.remove(us_case)

            count += 1
        
        return available_us