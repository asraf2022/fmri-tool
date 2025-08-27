from pathlib import Path
from typing import Dict
from fmritool.preprocess import run_preprocess
from fmritool.glm import run_first_level
from fmritool.viz import plot_glass, plot_stat_map

def run_pipeline(
    func: str,
    events: str,
    out_root: str = "runs",
    tr: float = 2.0,
    hrf: str = "spm",
    high_pass: float = 1/128,
    fwhm: float = 6.0,
    contrast: str = "condA - condB",
    viz_threshold: float = 3.0,
) -> Dict[str, str]:
    """
    Run end-to-end: preprocess (smooth) -> first-level GLM -> PNG visualizations.
    Returns a dict of output paths.
    """
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    # 1) preprocess
    preproc_path = Path(run_preprocess(func, out_dir=out_root, fwhm=fwhm))

    # 2) first-level GLM
    zmap_path = Path(
        run_first_level(
            func_img=str(preproc_path),
            events_tsv=events,
            out_dir=str(out),
            tr=tr,
            hrf=hrf,
            high_pass=high_pass,
            fwhm=fwhm,
            contrast=contrast,
        )
    )

    # 3) visualization
    glass_png = plot_glass(str(zmap_path), threshold=viz_threshold)
    stat_png  = plot_stat_map(str(zmap_path), threshold=viz_threshold)

    return {
        "preproc": str(preproc_path),
        "zmap": str(zmap_path),
        "glass_png": glass_png,
        "stat_png": stat_png,
    }
