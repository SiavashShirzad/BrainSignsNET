import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation

def get_inscribed_sphere_center(img, class_value):
    """
    Compute the 3D keypoint for a structure (given by class_value) using the 3D Euclidean distance transform.
    Returns (max_index, max_radius).
    """
    binary_mask = (img == class_value)
    if not np.any(binary_mask):
        return (np.nan, np.nan, np.nan), np.nan
    dt = distance_transform_edt(binary_mask)
    max_index = np.unravel_index(np.argmax(dt), dt.shape)
    max_radius = dt[max_index]
    return max_index, max_radius

def get_inscribed_circle_center_2d(img, class_value):
    """
    For an essentially 2D structure (e.g. CCPost or CCAnt), iterate over sagittal slices (x-axis)
    to select the slice with the largest area and compute the 2D Euclidean distance transform.
    Returns (keypoint, radius).
    """
    X, Y, Z = img.shape
    areas = np.zeros(X)
    for x in range(X):
        areas[x] = np.sum(img[x, :, :] == class_value)
    x_max = int(np.argmax(areas))
    if areas[x_max] == 0:
        return (np.nan, np.nan, np.nan), np.nan
    binary_slice = (img[x_max, :, :] == class_value)
    dt_2d = distance_transform_edt(binary_slice)
    idx_2d = np.unravel_index(np.argmax(dt_2d), dt_2d.shape)  # (y, z)
    radius = dt_2d[idx_2d]
    keypoint = (x_max, idx_2d[0], idx_2d[1])
    return keypoint, radius

def get_intraventricular_foramen(img, vent_value=14, rlv_value=43, llv_value=4):
    """
    Compute the keypoint (intraventricular foramen) for voxels in vent_value (e.g. 3rDVent)
    by finding the voxel that minimizes the sum of distances to voxels in rlv_value and llv_value.
    Returns (min_index, cost).
    """
    binary_vent = (img == vent_value)
    binary_rlv  = (img == rlv_value)
    binary_llv  = (img == llv_value)
    if not np.any(binary_vent) or not np.any(binary_rlv) or not np.any(binary_llv):
        return (np.nan, np.nan, np.nan), np.nan
    dist_rlv = distance_transform_edt(~binary_rlv)
    dist_llv = distance_transform_edt(~binary_llv)
    cost = np.full(img.shape, np.inf)
    cost[binary_vent] = dist_rlv[binary_vent] + dist_llv[binary_vent]
    min_index = np.unravel_index(np.argmin(cost), cost.shape)
    min_cost = cost[min_index]
    return min_index, min_cost

def get_inscribed_sphere_center_subregion(img, class_value, y_threshold, region='anterior'):
    """
    Restrict the search to a subregion based on the y-coordinate.
    If region=='anterior': use voxels with y > y_threshold.
    If region=='posterior': use voxels with y < y_threshold.
    Returns (max_index, max_radius) for that subregion.
    """
    binary_mask = (img == class_value)
    idx = np.indices(img.shape)
    if region == 'anterior':
        condition = idx[1] > y_threshold
    elif region == 'posterior':
        condition = idx[1] < y_threshold
    else:
        raise ValueError("region must be 'anterior' or 'posterior'")
    sub_mask = binary_mask & condition
    if not np.any(sub_mask):
        return (np.nan, np.nan, np.nan), np.nan
    dt = distance_transform_edt(sub_mask)
    max_index = np.unravel_index(np.argmax(dt), dt.shape)
    max_radius = dt[max_index]
    return max_index, max_radius

def get_custom_PC(img, brainstem_value=16, csf_value=24, target_value=14):
    """
    Custom keypoint "PC": From the union of BrainStem and CSF, select the voxel that is closest to target_value.
    Returns (min_index, distance).
    """
    union_mask = (img == brainstem_value) | (img == csf_value)
    target_mask = (img == target_value)
    if not np.any(union_mask) or not np.any(target_mask):
        return (np.nan, np.nan, np.nan), np.nan
    dt = distance_transform_edt(~target_mask)
    dt_union = np.where(union_mask, dt, np.inf)
    min_index = np.unravel_index(np.argmin(dt_union), dt_union.shape)
    min_val = dt_union[min_index]
    return min_index, min_val

def get_custom_AC(img, target_value=255, union_values=(28, 60)):
    """
    Custom keypoint "AC": From the union of LVDC and RVDC, select the voxel that is closest to target_value.
    Returns (min_index, distance).
    """
    union_mask = (img == union_values[0]) | (img == union_values[1])
    target_mask = (img == target_value)
    if not np.any(union_mask) or not np.any(target_mask):
        return (np.nan, np.nan, np.nan), np.nan
    dt = distance_transform_edt(~target_mask)
    dt_union = np.where(union_mask, dt, np.inf)
    min_index = np.unravel_index(np.argmin(dt_union), dt_union.shape)
    min_val = dt_union[min_index]
    return min_index, min_val

def get_LInternalCapsulGenu(img, target_labels=(10, 11, 13), source_value=2):
    """
    From the left white matter (mask value 2), compute the voxel minimizing the sum of distances
    to left thalamus, caudate, and pallidum.
    Returns (min_index, cost).
    """
    binary_source = (img == source_value)
    if not np.any(binary_source):
        return (np.nan, np.nan, np.nan), np.nan
    cost = np.zeros(img.shape, dtype=np.float32)
    for label in target_labels:
        target_mask = (img == label)
        if not np.any(target_mask):
            cost += np.inf
        else:
            dt = distance_transform_edt(~target_mask)
            cost += dt
    cost_source = np.where(binary_source, cost, np.inf)
    min_index = np.unravel_index(np.argmin(cost_source), cost_source.shape)
    min_cost = cost_source[min_index]
    return min_index, min_cost

def get_RInternalCapsulGenu(img, target_labels=(49, 50, 52), source_value=41):
    """
    From the right white matter (mask value 41), compute the voxel minimizing the sum of distances
    to right thalamus, caudate, and pallidum.
    Returns (min_index, cost).
    """
    binary_source = (img == source_value)
    if not np.any(binary_source):
        return (np.nan, np.nan, np.nan), np.nan
    cost = np.zeros(img.shape, dtype=np.float32)
    for label in target_labels:
        target_mask = (img == label)
        if not np.any(target_mask):
            cost += np.inf
        else:
            dt = distance_transform_edt(~target_mask)
            cost += dt
    cost_source = np.where(binary_source, cost, np.inf)
    min_index = np.unravel_index(np.argmin(cost_source), cost_source.shape)
    min_cost = cost_source[min_index]
    return min_index, min_cost

def get_internal_capsul_genu(img, target_labels, source_value):
    """
    Generic function to compute the internal capsule genu keypoint.
    Returns (min_index, cost).
    """
    binary_source = (img == source_value)
    if not np.any(binary_source):
        return (np.nan, np.nan, np.nan), np.nan
    cost = np.zeros(img.shape, dtype=np.float32)
    for label in target_labels:
        target_mask = (img == label)
        if not np.any(target_mask):
            cost += np.inf
        else:
            dt = distance_transform_edt(~target_mask)
            cost += dt
    cost_source = np.where(binary_source, cost, np.inf)
    min_index = np.unravel_index(np.argmin(cost_source), cost_source.shape)
    min_cost = cost_source[min_index]
    return min_index, min_cost

def normalize_coordinate(coord, shape):
    """
    Normalize a 3D coordinate based on the provided shape.
    """
    if coord is None or any(np.isnan(coord)):
        return [np.nan, np.nan, np.nan]
    return [coord[0] / shape[0], coord[1] / shape[1], coord[2] / shape[2]]
