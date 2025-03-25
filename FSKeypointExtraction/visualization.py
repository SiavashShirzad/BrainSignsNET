import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import nibabel as nib
from skimage import measure
import plotly.graph_objects as go
import plotly.io as pio

# Optionally set the Plotly renderer (e.g. for Jupyter notebooks)
pio.renderers.default = "notebook"

def visualize_keypoints_matplotlib(t1, mask, centers, radii, class_dict):
    """
    Visualize keypoints overlaid on the scan using axial, sagittal, and coronal views.
    'centers' and 'radii' are dictionaries keyed by structure name.
    'class_dict' maps structure names to mask values.
    """
    all_keys = list(centers.keys())
    fig, axes = plt.subplots(len(all_keys), 3, figsize=(15, 5 * len(all_keys)))
    if len(all_keys) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, key in enumerate(all_keys):
        kp = centers.get(key, (np.nan, np.nan, np.nan))
        try:
            x, y, z = map(int, np.round(kp))
        except Exception:
            x, y, z = (np.nan, np.nan, np.nan)
        rad = radii.get(key, np.nan)
        
        # Determine overlay label
        if key in ["PC"]:
            overlay_label = 16
        elif key in ["AC"]:
            overlay_label = 28
        elif key in ["Midbrain", "Pons", "Medulla"]:
            overlay_label = 16
        elif key == "LInternalCapsulGenu":
            overlay_label = 2
        elif key == "RInternalCapsulGenu":
            overlay_label = 41
        else:
            overlay_label = class_dict.get(key, 0)
        mask_overlay = (mask == overlay_label)
        
        # Axial view (slice along z)
        ax0 = axes[i, 0]
        if not np.isnan(z) and 0 <= z < t1.shape[2]:
            axial = t1[:, :, z]
            ax0.imshow(axial, cmap='gray', origin='lower')
            ax0.imshow(mask_overlay[:, :, z], cmap='Reds', alpha=0.3, origin='lower')
            ax0.scatter(y, x, c='blue', s=50)
            ax0.set_title(f"{key} Axial (z={z}, r={rad:.2f})")
        else:
            ax0.text(0.5, 0.5, "z index out of bounds", ha='center', va='center')
            ax0.set_title(f"{key} Axial")
        ax0.axis('off')
        
        # Sagittal view (slice along x)
        ax1 = axes[i, 1]
        if not np.isnan(x) and 0 <= x < t1.shape[0]:
            sagittal = t1[x, :, :]
            ax1.imshow(sagittal, cmap='gray', origin='lower')
            ax1.imshow(mask_overlay[x, :, :], cmap='Reds', alpha=0.3, origin='lower')
            ax1.scatter(z, y, c='blue', s=50)
            ax1.set_title(f"{key} Sagittal (x={x}, r={rad:.2f})")
        else:
            ax1.text(0.5, 0.5, "x index out of bounds", ha='center', va='center')
            ax1.set_title(f"{key} Sagittal")
        ax1.axis('off')
        
        # Coronal view (slice along y)
        ax2 = axes[i, 2]
        if not np.isnan(y) and 0 <= y < t1.shape[1]:
            coronal = t1[:, y, :]
            ax2.imshow(coronal, cmap='gray', origin='lower')
            ax2.imshow(mask_overlay[:, y, :], cmap='Reds', alpha=0.3, origin='lower')
            ax2.scatter(z, x, c='blue', s=50)
            ax2.set_title(f"{key} Coronal (y={y}, r={rad:.2f})")
        else:
            ax2.text(0.5, 0.5, "y index out of bounds", ha='center', va='center')
            ax2.set_title(f"{key} Coronal")
        ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 3D Scatter Plot of Keypoints
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    for key in all_keys:
        kp = centers.get(key, (np.nan, np.nan, np.nan))
        if not np.isnan(kp[0]):
            x, y, z = kp
            ax_3d.scatter(x, y, z, label=key, s=50)
    ax_3d.set_xlabel('X (Right)')
    ax_3d.set_ylabel('Y (Anterior)')
    ax_3d.set_zlabel('Z (Superior)')
    ax_3d.legend()
    plt.title("3D Scatter Plot of Keypoints")
    plt.show()

def visualize_roi_with_sphere_plotly(mask_data, roi_value, center_index, max_radius):
    """
    Visualize an ROI surface mesh (e.g. RThalamus) along with the maximal inscribed sphere using Plotly.
    """
    # Extract ROI and compute surface mesh using marching cubes
    roi = (mask_data == roi_value)
    roi_float = roi.astype(np.float32)
    verts, faces, normals, values = measure.marching_cubes(roi_float, level=0.5)
    x_roi, y_roi, z_roi = verts[:, 0], verts[:, 1], verts[:, 2]
    i_roi, j_roi, k_roi = faces[:, 0], faces[:, 1], faces[:, 2]
    
    # Create sphere surface coordinates
    n_u, n_v = 50, 50
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    u, v = np.meshgrid(u, v)
    x_sphere = center_index[0] + max_radius * np.sin(v) * np.cos(u)
    y_sphere = center_index[1] + max_radius * np.sin(v) * np.sin(u)
    z_sphere = center_index[2] + max_radius * np.cos(v)
    
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=x_roi, y=y_roi, z=z_roi,
        i=i_roi, j=j_roi, k=k_roi,
        color='lightblue', opacity=0.5,
        name='ROI Surface'
    ))
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale='Reds', opacity=0.6,
        name='Maximal Inscribed Sphere',
        showscale=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[center_index[0]], y=[center_index[1]], z=[center_index[2]],
        mode='markers',
        marker=dict(size=5, color='black'),
        name='Sphere Center'
    ))
    fig.update_layout(
        title="Maximal Inscribed Sphere in ROI",
        scene=dict(
            xaxis_title="X (voxels)",
            yaxis_title="Y (voxels)",
            zaxis_title="Z (voxels)"
        ),
        width=800, height=700
    )
    fig.show()

def visualize_iv_foramen_plotly(mask_data):
    """
    Visualize the interventricular foramen and related structures using Plotly.
    This function computes the keypoint (and nearest points) and then plots the corresponding meshes.
    """
    from utils import get_intraventricular_foramen
    # Compute the keypoint for the interventricular foramen
    iv_foramen, cost = get_intraventricular_foramen(mask_data, vent_value=14, rlv_value=43, llv_value=4)
    
    # Compute nearest voxels for RLV and LLV
    binary_rlv = (mask_data == 43)
    binary_llv = (mask_data == 4)
    dt_rlv, indices_rlv = distance_transform_edt(~binary_rlv, return_indices=True)
    dt_llv, indices_llv = distance_transform_edt(~binary_llv, return_indices=True)
    nearest_rlv = (indices_rlv[0][iv_foramen], indices_rlv[1][iv_foramen], indices_rlv[2][iv_foramen])
    nearest_llv = (indices_llv[0][iv_foramen], indices_llv[1][iv_foramen], indices_llv[2][iv_foramen])
    
    # Extract surfaces using marching cubes
    roi_vent = (mask_data == 14)
    verts_vent, faces_vent, _, _ = measure.marching_cubes(roi_vent.astype(np.float32), level=0.5)
    x_vent, y_vent, z_vent = verts_vent[:, 0], verts_vent[:, 1], verts_vent[:, 2]
    i_vent, j_vent, k_vent = faces_vent[:, 0], faces_vent[:, 1], faces_vent[:, 2]
    
    roi_rlv = (mask_data == 43)
    verts_rlv, faces_rlv, _, _ = measure.marching_cubes(roi_rlv.astype(np.float32), level=0.5)
    x_rlv, y_rlv, z_rlv = verts_rlv[:, 0], verts_rlv[:, 1], verts_rlv[:, 2]
    i_rlv, j_rlv, k_rlv = faces_rlv[:, 0], faces_rlv[:, 1], faces_rlv[:, 2]
    
    roi_llv = (mask_data == 4)
    verts_llv, faces_llv, _, _ = measure.marching_cubes(roi_llv.astype(np.float32), level=0.5)
    x_llv, y_llv, z_llv = verts_llv[:, 0], verts_llv[:, 1], verts_llv[:, 2]
    i_llv, j_llv, k_llv = faces_llv[:, 0], faces_llv[:, 1], faces_llv[:, 2]
    
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=x_vent, y=y_vent, z=z_vent,
        i=i_vent, j=j_vent, k=k_vent,
        color='lightblue',
        opacity=0.8,
        name='3rd Ventricle'
    ))
    fig.add_trace(go.Mesh3d(
        x=x_rlv, y=y_rlv, z=z_rlv,
        i=i_rlv, j=j_rlv, k=k_rlv,
        color='lightgreen',
        opacity=0.5,
        name='Right Lateral Ventricle'
    ))
    fig.add_trace(go.Mesh3d(
        x=x_llv, y=y_llv, z=z_llv,
        i=i_llv, j=j_llv, k=k_llv,
        color='lightyellow',
        opacity=0.5,
        name='Left Lateral Ventricle'
    ))
    fig.add_trace(go.Scatter3d(
        x=[iv_foramen[0]], y=[iv_foramen[1]], z=[iv_foramen[2]],
        mode='markers+text',
        marker=dict(size=10, color='red'),
        name="Interventricular Foramen"
    ))
    fig.add_trace(go.Scatter3d(
        x=[nearest_rlv[0]], y=[nearest_rlv[1]], z=[nearest_rlv[2]],
        mode='markers+text',
        marker=dict(size=10, color='purple'),
        name="Nearest RLV"
    ))
    fig.add_trace(go.Scatter3d(
        x=[nearest_llv[0]], y=[nearest_llv[1]], z=[nearest_llv[2]],
        mode='markers+text',
        marker=dict(size=10, color='orange'),
        name="Nearest LLV"
    ))
    fig.add_trace(go.Scatter3d(
        x=[iv_foramen[0], nearest_rlv[0]],
        y=[iv_foramen[1], nearest_rlv[1]],
        z=[iv_foramen[2], nearest_rlv[2]],
        mode='lines',
        line=dict(color='black', width=4),
        name="Distance to RLV"
    ))
    fig.add_trace(go.Scatter3d(
        x=[iv_foramen[0], nearest_llv[0]],
        y=[iv_foramen[1], nearest_llv[1]],
        z=[iv_foramen[2], nearest_llv[2]],
        mode='lines',
        line=dict(color='black', width=4),
        name="Distance to LLV"
    ))
    fig.update_layout(
        title="Interventricular Foramen Visualization",
        width=1000,
        height=1000
    )
    fig.show()

if __name__ == "__main__":
    # Example usage for Matplotlib visualization:
    # Replace the following paths with your own T1 and mask file paths.
    t1_file = 'path/to/T1.mgz'
    mask_file = 'path/to/aparc+aseg.mgz'
    t1_img = nib.load(t1_file)
    t1_data = t1_img.get_fdata()
    mask_img = nib.load(mask_file)
    mask_img = nib.as_closest_canonical(mask_img)
    mask_data = mask_img.get_fdata()
    
    # Dummy keypoints for demonstration (in practice, compute these using extraction routines)
    centers = {
        'CCPost': (30, 40, 50),
        'CCAnt': (35, 45, 55),
        'Midbrain': (25, 35, 45)
    }
    radii = {
        'CCPost': 5,
        'CCAnt': 6,
        'Midbrain': 4
    }
    # Standard structure mapping (example)
    class_dict = {'CCPost': 251, 'CCAnt': 255, 'Midbrain': 16}
    
    visualize_keypoints_matplotlib(t1_data, mask_data, centers, radii, class_dict)
