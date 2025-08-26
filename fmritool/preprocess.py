from pathlib import Path
from nilearn import image

def run_preprocess(func_path: str, out_dir: str = "derivatives", fwhm: float = 6.0) -> str:
    """
    Minimal 'preprocessing' so the pipeline runs anywhere:
    - load functional 4D NIfTI
    - apply Gaussian smoothing (FWHM in mm)
    - save to derivatives/preproc_<originalname>
    """
    func_path = Path(func_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = image.load_img(str(func_path))
    smoothed = image.smooth_img(img, fwhm)

    out_file = out / f"preproc_{func_path.name}"
    smoothed.to_filename(str(out_file))
    return str(out_file)
