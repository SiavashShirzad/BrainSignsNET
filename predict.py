"""Landmark prediction on raw T1-weighted NIfTI scans with the released BrainSignsNET weights.
No skull stripping, registration or cropping is needed. Output coordinates are in scanner (RAS) space, in mm.

  python predict.py scan1.nii.gz [scan2.nii.gz ...] -o landmarks.csv
  python predict.py --list scans.csv -o landmarks.csv      # scans.csv has columns id,image_path
Options: --weights (default weights/brainsignsnet_fp16.pth), --device (default cuda if available)."""
import os, argparse
import numpy as np, pandas as pd, nibabel as nib, torch, torch.nn.functional as F
from model import load, LANDMARKS

VOL = 128
HERE = os.path.dirname(os.path.abspath(__file__))


def preprocess(path):
    """Canonical (RAS) orientation; the largest physical dimension is resampled to 128 voxels, the others keep the same
    voxel size and are centre-padded; z-score intensity normalization."""
    img = nib.as_closest_canonical(nib.load(path))
    data = np.ascontiguousarray(img.get_fdata(dtype=np.float32))
    data = np.ascontiguousarray(data[..., 0]) if data.ndim == 4 else data
    sp = np.array(img.header.get_zooms()[:3], float); phys = np.array(data.shape) * sp
    ts = phys.max() / VOL; new = tuple(int(round(p / ts)) for p in phys)
    v = F.interpolate(torch.from_numpy(data)[None, None], size=new, mode="trilinear", align_corners=False)
    pv = np.array([VOL] * 3) - np.array(new)
    pad = (pv[2] // 2, pv[2] - pv[2] // 2, pv[1] // 2, pv[1] - pv[1] // 2, pv[0] // 2, pv[0] - pv[0] // 2)
    v = F.pad(v, pad)[0]
    s = v.std(); v = (v - v.mean()) / (s if s > 1e-6 else 1.0)
    off = np.array([pad[4], pad[2], pad[0]])
    return v[None], img, ts, off, sp


@torch.no_grad()
def predict(model, path, device):
    x, img, ts, off, sp = preprocess(path)
    g = model(x.to(device))[2][0].view(-1, 3).cpu().numpy()     # grid coordinates (K, 3)
    vox = (g - off) * ts / sp                                   # canonical voxel coordinates
    return nib.affines.apply_affine(img.affine, vox)           # scanner RAS (mm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scans", nargs="*"); ap.add_argument("--list"); ap.add_argument("-o", "--out", default="landmarks.csv")
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "brainsignsnet_fp16.pth"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    df = pd.read_csv(a.list) if a.list else pd.DataFrame(dict(id=[os.path.basename(s) for s in a.scans], image_path=a.scans))
    model = load(a.weights, a.device); rows = []
    for r in df.itertuples():
        try:
            w = predict(model, r.image_path, a.device)
        except Exception as e:
            print(f"failed: {r.id}: {e}"); continue
        rows += [dict(id=r.id, landmark=n, x=p[0], y=p[1], z=p[2]) for n, p in zip(LANDMARKS, w)]
    pd.DataFrame(rows).to_csv(a.out, index=False); print(f"{len(rows) // len(LANDMARKS)} of {len(df)} scans -> {a.out}")
