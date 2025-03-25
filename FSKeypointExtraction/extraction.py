import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from tqdm import tqdm
import concurrent.futures

from utils import (
    get_inscribed_sphere_center,
    get_inscribed_circle_center_2d,
    get_intraventricular_foramen,
    get_inscribed_sphere_center_subregion,
    get_custom_PC,
    get_custom_AC,
    get_internal_capsul_genu,
    normalize_coordinate
)

def process_mask(row):
    """
    Process a single mask file: load, reorient, extract keypoints, and normalize coordinates.
    Expects 'row' (from a DataFrame) to have keys 'id' and 'path'.
    """
    result = {}
    result['id'] = row['id']
    result['path'] = row['path']
    try:
        mask_path = row['path']
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"File not found: {mask_path}")
        m_img = nib.load(mask_path)
        m_img = nib.as_closest_canonical(m_img)
        mask_data = m_img.get_fdata()
        shape = mask_data.shape

        # Standard structure dictionary
        class_dict = {
            'CCPost':      251,
            'CCAnt':       255,
            'RPutamen':     51,
            'LPutamen':     12,
            'LCaudate':     11,
            'LThalamus':    10,
            'LPallidum':    13,
            'LHippocampus': 17,
            'LAmygdala':    18,
            'RCaudate':     50,
            'RThalamus':    49,
            'RPallidum':    52,
            'RHippocampus': 53,
            'RAmygdala':    54,
            'OpticChiasm':  85
        }
        centers = {}
        radii = {}
        # Extract standard keypoints (use 2D extraction for CCPost/CCAnt)
        for structure, value in class_dict.items():
            if structure in ['CCPost', 'CCAnt']:
                kp, rad = get_inscribed_circle_center_2d(mask_data, value)
            else:
                kp, rad = get_inscribed_sphere_center(mask_data, value)
            centers[structure] = kp
            radii[structure] = rad

        # Custom key: IntraventricularForamen
        inv_kp, inv_cost = get_intraventricular_foramen(mask_data, vent_value=14, rlv_value=43, llv_value=4)
        centers["IntraventricularForamen"] = inv_kp
        radii["IntraventricularForamen"] = inv_cost

        # BrainStem splitting using cerebellar connection
        brainstem_mask = (mask_data == 16)
        cerebellum_mask = ((mask_data == 7) | (mask_data == 46))
        struct_elem = np.ones((3, 3, 3), dtype=bool)
        cereb_dilated = binary_dilation(cerebellum_mask, structure=struct_elem, iterations=2)
        connection = brainstem_mask & cereb_dilated
        if np.any(connection):
            inds = np.indices(mask_data.shape)
            z_map = inds[2]
            z_conn = z_map[connection]
            z_min = np.min(z_conn)
            z_max = np.max(z_conn)
        else:
            z_min = np.nan
            z_max = np.nan

        medulla_mask = brainstem_mask & (z_map < z_min)
        pons_mask = brainstem_mask & ((z_map >= z_min) & (z_map <= z_max))
        midbrain_mask = brainstem_mask & (z_map > z_max)

        mid_kp, mid_rad = get_inscribed_sphere_center(midbrain_mask.astype(np.float32), 1)
        pons_kp, pons_rad = get_inscribed_sphere_center(pons_mask.astype(np.float32), 1)
        medulla_kp, medulla_rad = get_inscribed_sphere_center(medulla_mask.astype(np.float32), 1)
        centers["Midbrain"] = mid_kp
        radii["Midbrain"] = mid_rad
        centers["Pons"] = pons_kp
        radii["Pons"] = pons_rad
        centers["Medulla"] = medulla_kp
        radii["Medulla"] = medulla_rad

        # Custom keys PC and AC
        pc_kp, pc_val = get_custom_PC(mask_data, brainstem_value=16, csf_value=24, target_value=14)
        centers["PC"] = pc_kp
        radii["PC"] = pc_val

        ac_kp, ac_val = get_custom_AC(mask_data, target_value=255, union_values=(28, 60))
        centers["AC"] = ac_kp
        radii["AC"] = ac_val

        # Custom keys for Internal Capsule Genu (left and right)
        l_icg_kp, l_icg_cost = get_internal_capsul_genu(mask_data, target_labels=(10, 11, 13, 28), source_value=2)
        centers["LInternalCapsulGenu"] = l_icg_kp
        radii["LInternalCapsulGenu"] = l_icg_cost

        r_icg_kp, r_icg_cost = get_internal_capsul_genu(mask_data, target_labels=(49, 50, 52, 60), source_value=41)
        centers["RInternalCapsulGenu"] = r_icg_kp
        radii["RInternalCapsulGenu"] = r_icg_cost

        # Normalize coordinates based on mask shape.
        result_data = {'id': row['id'], 'path': row['path']}
        all_keys = list(class_dict.keys()) + ["IntraventricularForamen", "Midbrain", "Pons", "Medulla",
                                                "PC", "AC", "LInternalCapsulGenu", "RInternalCapsulGenu"]
        for key in all_keys:
            result_data[key + '_center'] = normalize_coordinate(centers.get(key, (np.nan, np.nan, np.nan)), shape)
            # For standard keys and brainstem partitions, store max_radius; for custom keys, leave as NaN.
            if key not in ["PC", "AC", "LInternalCapsulGenu", "RInternalCapsulGenu"]:
                result_data[key + '_max_radius'] = radii.get(key, np.nan)
            else:
                result_data[key + '_max_radius'] = np.nan
        result_data['error'] = ""
        result = result_data
    except Exception as e:
        result = {'id': row['id'], 'path': row['path'], 'error': str(e)}
    return result

def process_scans(df_scans, n_workers=4):
    """
    Process multiple scans using multiprocessing.
    df_scans should be a pandas DataFrame with columns 'id' and 'path'.
    Returns a DataFrame with the extracted keypoints.
    """
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_mask, row): row for _, row in df_scans.iterrows()}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing scans"):
            try:
                res = future.result()
                results.append(res)
            except Exception as exc:
                row = futures[future]
                results.append({'id': row['id'], 'path': row['path'], 'error': str(exc)})
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Example usage: create a dummy DataFrame and process scans.
    df_example = pd.DataFrame([
        {'id': 'scan1', 'path': 'path/to/your/mask_file.mgz'}
    ])
    df_results = process_scans(df_example, n_workers=1)
    print(df_results)
