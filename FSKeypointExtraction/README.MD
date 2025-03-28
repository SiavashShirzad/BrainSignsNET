# MRI Keypoint Extraction and Visualization Toolkit

This repository contains Python modules designed for extracting anatomical keypoints from 3D MRI segmentation masks, and visualizing them through interactive 2D and 3D plots.

The code is structured into three main Python modules:

- **`utils.py`**: Contains core utility functions for computing keypoints from segmentation masks.
- **`extraction.py`**: Handles the extraction of keypoints from MRI segmentation masks, supporting parallel batch processing.
- **`visualization.py`**: Provides interactive and static visualization capabilities for keypoints using Matplotlib and Plotly.

---

## 🛠️ Installation

### Step 1: Clone the repository

```bash
git clone <your_repository_url>
cd <repository_folder>
```
### Step 2: Set up the Python environment

# We recommend creating a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### Step 3: Install dependencies

# Install required packages using pip:

```bash
pip install -r requirements.txt
```
### 📚 Usage
## Extract Keypoints (Single Scan)

# Use extraction.py to process individual segmentation masks:

```bash
from extraction import process_mask
import pandas as pd

row = {
    'id': 'subject01',
    'path': '/path/to/your/segmentation_mask.mgz'
}

results = process_mask(row)
print(results)
```

### Batch Processing (Parallelized Extraction)

# If you have a dataframe of scans:

```bash
from extraction import process_scans
import pandas as pd

df_scans = pd.DataFrame([
    {'id': 'subj01', 'path': '/path/to/mask01.mgz'},
    {'id': 'subj02', 'path': '/path/to/mask02.mgz'},
    # ... more scans
])

df_results = process_scans(df_scans, n_workers=8)
df_results.to_csv('extracted_keypoints.csv', index=False)
```

### Visualization Matplotlib (2D and 3D)

#Visualize keypoints and anatomical regions in static images:

```bash
from visualization import visualize_keypoints_matplotlib
import nibabel as nib

t1_data = nib.load('path/to/T1.mgz').get_fdata()
mask_data = nib.load('path/to/aparc+aseg.mgz').get_fdata()

centers = {'CCPost': (30, 40, 50), 'Midbrain': (35, 45, 60)}
radii = {'CCPost': 5, 'Midbrain': 7}
class_dict = {'CCPost': 251, 'Midbrain': 16}

visualize_keypoints_matplotlib(t1_data, mask_data, centers, radii, class_dict)
```

### Plotly (Interactive 3D Visualization)

# Interactive visualization of specific regions and keypoints:
```bash
from visualization import visualize_roi_with_sphere_plotly, visualize_iv_foramen_plotly
import nibabel as nib

mask_data = nib.load('path/to/aparc+aseg.mgz').get_fdata()

# Visualize ROI with sphere
visualize_roi_with_sphere_plotly(mask_data, roi_value=49, center_index=(50, 60, 70), max_radius=15)

# Visualize Interventricular Foramen
visualize_iv_foramen_plotly(mask_data)
```


### 📄 Citation

If you use this toolkit in your research or applications, please cite accordingly: