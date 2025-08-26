from pathlib import Path
from nilearn import plotting, image

def plot_glass(zmap_path: str, out_file: str = None, threshold: float = 3.0) -> str:
    """
    Make a glass brain plot from a Z-map NIfTI.
    Returns the path to the saved PNG.
    """
    zmap = image.load_img(zmap_path)

    if out_file is None:
        out_file = str(Path(zmap_path).with_suffix("")) + "_glass.png"

    display = plotting.plot_glass_brain(
        zmap, threshold=threshold, colorbar=True, plot_abs=False, display_mode="lyrz"
    )
    display.savefig(out_file)
    display.close()
    return out_file

def plot_stat_map(zmap_path: str, bg_img: str = None, out_file: str = None, threshold: float = 3.0) -> str:
    """
    Make a slice overlay plot of a Z-map on a background (if provided).
    """
    zmap = image.load_img(zmap_path)

    if out_file is None:
        out_file = str(Path(zmap_path).with_suffix("")) + "_stat.png"

    display = plotting.plot_stat_map(
        zmap, bg_img=bg_img, threshold=threshold, display_mode="ortho", colorbar=True
    )
    display.savefig(out_file)
    display.close()
    return out_file
