from pathlib import Path
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel

def run_first_level(
    func_img: str,
    events_tsv: str,
    out_dir: str = "first_level",
    tr: float = 2.0,
    hrf: str = "spm",            # "spm" or "glover"
    high_pass: float = 1/128,    # Hz (≈0.0078)
    fwhm: float = 6.0,
    contrast: str = "condA - condB",
) -> str:
    """
    Fit a simple first-level GLM and compute one contrast.
    events_tsv must have: onset, duration, trial_type.
    Returns path to the saved Z-map.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(events_tsv, sep="\t")
    model = FirstLevelModel(
        t_r=tr,
        hrf_model=hrf,
        drift_model="cosine",
        high_pass=high_pass,
        smoothing_fwhm=fwhm,
        noise_model="ar1",
        standardize=False,
        minimize_memory=True,
    )
    fitted = model.fit(func_img, events=events)

    z_map = fitted.compute_contrast(contrast, output_type="z_score")
    out_file = out / f"zmap_{contrast.replace(' ', '')}.nii.gz"
    z_map.to_filename(str(out_file))
    return str(out_file)
