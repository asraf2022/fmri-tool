from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from nilearn import image, masking, signal, datasets, plotting
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker, NiftiSpheresMasker

def rs_clean(
    func_path: str,
    out_dir: str = "rs_derivatives",
    tr: float = 2.0,
    bandpass: tuple[float, float] = (0.01, 0.1),
    detrend: bool = True,
    standardize: str = "zscore",
    confounds_tsv: str | None = None,
) -> str:
    """
    Minimal rs-fMRI cleaning:
      - optional nuisance regression (confounds_tsv columns if given)
      - detrending
      - band-pass filtering
      - voxel-wise standardization
    Returns path to cleaned 4D NIfTI.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    img = image.load_img(func_path)
    X = masking.apply_mask(img, masking.compute_epi_mask(img))  # shape: time x voxels

    confounds = None
    if confounds_tsv:
        df = pd.read_csv(confounds_tsv, sep="\t")
        # pick common motion confounds if present; otherwise use all numeric
        cols = [c for c in df.columns if c.lower() in
                ["trans_x","trans_y","trans_z","rot_x","rot_y","rot_z","framewise_displacement","dvars"]]
        if not cols:
            cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        confounds = df[cols].values

    cleaned = signal.clean(
        X,
        t_r=tr,
        detrend=detrend,
        standardize=standardize,
        low_pass=bandpass[1],
        high_pass=bandpass[0],
        confounds=confounds,
    )
    cleaned_img = masking.unmask(cleaned, masking.compute_epi_mask(img))
    out_file = str(out / f"cleaned_{Path(func_path).name}")
    cleaned_img.to_filename(out_file)
    return out_file

def rs_timeseries(
    func_img: str,
    atlas: str = "schaefer_100p_7n",
    tr: float = 2.0,
    out_csv: str = "rs_timeseries.csv",
) -> str:
    """
    Extract parcel-wise timeseries using a standard atlas.
    """
    if atlas == "schaefer_100p_7n":
        sch = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
        labels_img = sch.maps
        labels = sch.labels
    elif atlas == "harvard_oxford_cort_maxprob":
        ho = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        labels_img = ho.maps
        labels = ho.labels
    else:
        raise ValueError("Unknown atlas")

    masker = NiftiLabelsMasker(labels_img=labels_img, standardize=True, t_r=tr)
    ts = masker.fit_transform(func_img)  # shape: time x parcels
    df = pd.DataFrame(ts, columns=[str(l) for l in labels])
    df.to_csv(out_csv, index=False)
    return out_csv

def rs_connectivity(
    timeseries_csv: str,
    kind: tuple[str, ...] = ("correlation", "partial correlation"),
    out_prefix: str = "rs_conn",
) -> list[str]:
    """
    Compute connectivity matrices from parcel timeseries and save PNG + CSV for each kind.
    """
    ts = pd.read_csv(timeseries_csv).values
    outs: list[str] = []
    for k in kind:
        cm = ConnectivityMeasure(kind="tangent" if k == "tangent" else k)
        mat = cm.fit_transform([ts])[0]  # square matrix

        # Save CSV
        csv_path = f"{out_prefix}_{k.replace(' ', '_')}.csv"
        pd.DataFrame(mat).to_csv(csv_path, index=False)

        # Save PNG
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mat, interpolation="nearest")
        plt.title(f"{k} connectivity")
        plt.colorbar()
        png_path = f"{out_prefix}_{k.replace(' ', '_')}.png"
        plt.savefig(png_path, bbox_inches="tight", dpi=150)
        plt.close()

        outs += [csv_path, png_path]
    return outs

def rs_seedmap(
    func_img: str,
    seed_mni: tuple[float, float, float],
    radius_mm: float = 6.0,
    tr: float = 2.0,
    out_nii: str = "rs_seed_zmap.nii.gz",
    out_png: str = "rs_seed_zmap.png",
) -> tuple[str, str]:
    """
    Seed-based correlation map from MNI coords; returns NIfTI + PNG.
    """
    masker = NiftiSpheresMasker([seed_mni], radius=radius_mm, detrend=True, standardize=True, t_r=tr)
    seed_ts = masker.fit_transform(func_img)[:, 0]  # (time,)

    # Whole-brain timeseries for each voxel
    brain_mask = masking.compute_epi_mask(image.load_img(func_img))
    all_ts = masking.apply_mask(func_img, brain_mask)  # time x voxels

    # Correlate seed with each voxel
    seed_ts = (seed_ts - seed_ts.mean()) / (seed_ts.std() + 1e-8)
    all_ts = (all_ts - all_ts.mean(0)) / (all_ts.std(0) + 1e-8)
    corr = all_ts.T @ seed_ts / all_ts.shape[0]  # voxels
    # Fisher z-transform
    z = np.arctanh(np.clip(corr, -0.999999, 0.999999))

    z_img = masking.unmask(z, brain_mask)
    z_img.to_filename(out_nii)

    # PNG
    disp = plotting.plot_glass_brain(z_img, colorbar=True, display_mode="lyrz", threshold=0.3)
    disp.savefig(out_png)
    disp.close()
    return out_nii, out_png
