import typer
from fmritool.preprocess import run_preprocess
from fmritool.glm import run_first_level
from pathlib import Path

app = typer.Typer(help="fmritool CLI")

@app.command()
def hello(name: str = "world"):
    """Say hello (sanity-check command)."""
    print(f"hello {name}")

@app.command()
def prep(
    func: str = typer.Option(..., help="Path to functional 4D NIfTI (.nii/.nii.gz)"),
    out: str = typer.Option("derivatives", help="Output folder"),
    fwhm: float = typer.Option(6.0, help="Smoothing FWHM (mm)"),
):
    """Minimal fallback preprocessing: smooth the input NIfTI."""
    out_file = run_preprocess(func_path=func, out_dir=out, fwhm=fwhm)
    print(f"Saved preprocessed file: {out_file}")

@app.command("make-demo")
def make_demo(
    out_dir: str = typer.Option("examples", help="Where to place demo files"),
    tr: float = typer.Option(2.0, help="Repetition time (s) for info only"),
    n_vols: int = typer.Option(40, help="Number of timepoints (volumes)"),
):
    """Create a tiny synthetic 4D NIfTI + events.tsv for testing."""
    import numpy as np, nibabel as nib
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    shape = (64, 64, 10, n_vols)
    data = np.random.default_rng(42).normal(0, 1, size=shape).astype("float32")
    img = nib.Nifti1Image(data, affine=np.eye(4))
    func_path = out / "demo_func.nii.gz"; nib.save(img, str(func_path))
    events_path = out / "events.tsv"
    events_path.write_text(
        "onset\tduration\ttrial_type\n"
        "0\t2\tcondA\n20\t2\tcondB\n40\t2\tcondA\n60\t2\tcondB\n"
    )
    print(f"Demo functional: {func_path}\nDemo events:     {events_path}\n(TR: {tr}s)")

@app.command("first-level")
def first_level(
    func: str = typer.Option(..., help="Path to (preprocessed) NIfTI"),
    events: str = typer.Option(..., help="Path to events.tsv (onset,duration,trial_type)"),
    out: str = typer.Option("first_level", help="Output folder"),
    tr: float = typer.Option(2.0, help="TR in seconds"),
    hrf: str = typer.Option("spm", help="HRF model: spm|glover"),
    high_pass: float = typer.Option(1/128, help="High-pass (Hz), e.g., 0.0078"),
    fwhm: float = typer.Option(6.0, help="Smoothing FWHM (mm)"),
    contrast: str = typer.Option("condA - condB", help="Contrast expression"),
):
    """Fit a first-level GLM and save a Z-map."""
    zmap_path = run_first_level(
        func_img=func, events_tsv=events, out_dir=out,
        tr=tr, hrf=hrf, high_pass=high_pass, fwhm=fwhm, contrast=contrast
    )
    print(f"Saved Z-map: {zmap_path}")


from fmritool import viz

@app.command("viz")
def visualize(
    zmap: str = typer.Option(..., help="Path to a Z-map NIfTI"),
    threshold: float = typer.Option(3.0, help="Z threshold for plotting"),
):
    """Make glass brain + stat map PNGs from a Z-map."""
    glass = viz.plot_glass(zmap, threshold=threshold)
    stat = viz.plot_stat_map(zmap, threshold=threshold)
    print(f"Saved glass brain: {glass}")
    print(f"Saved stat map:    {stat}")

from fmritool.pipeline import run_pipeline

@app.command()
def pipeline(
    func: str = typer.Option(..., help="Path to raw functional 4D NIfTI"),
    events: str = typer.Option(..., help="Path to events.tsv (onset,duration,trial_type)"),
    out: str = typer.Option("runs", help="Output root folder"),
    tr: float = typer.Option(2.0, help="TR (s)"),
    hrf: str = typer.Option("spm", help="HRF model: spm|glover"),
    high_pass: float = typer.Option(1/128, help="High-pass (Hz), e.g., 0.0078"),
    fwhm: float = typer.Option(6.0, help="Smoothing FWHM (mm)"),
    contrast: str = typer.Option("condA - condB", help="Contrast expression, e.g., 'condA - condB'"),
    viz_threshold: float = typer.Option(3.0, help="Z threshold for plotting"),
):
    """
    Run preprocess -> first-level GLM -> PNG visualization in one go.
    """
    paths = run_pipeline(
        func=func,
        events=events,
        out_root=out,
        tr=tr,
        hrf=hrf,
        high_pass=high_pass,
        fwhm=fwhm,
        contrast=contrast,
        viz_threshold=viz_threshold,
    )
    for k, v in paths.items():
        print(f"{k}: {v}")

from fmritool.io_utils import find_bids_runs
from fmritool.pipeline import run_pipeline

@app.command("bids-pipeline")
def bids_pipeline(
    bids: str = typer.Option(..., help="Path to BIDS root"),
    subject: str = typer.Option(None, help='Subject label, e.g., "01" (omit "sub-")'),
    task: str = typer.Option(None, help='Task label, e.g., "faces"'),
    out: str = typer.Option("runs", help="Output root folder"),
    tr: float = typer.Option(2.0, help="TR (s)"),
    hrf: str = typer.Option("spm", help="HRF model: spm|glover"),
    high_pass: float = typer.Option(1/128, help="High-pass (Hz)"),
    fwhm: float = typer.Option(6.0, help="Smoothing FWHM (mm)"),
    contrast: str = typer.Option("condA - condB", help="Contrast expression"),
    viz_threshold: float = typer.Option(3.0, help="Z threshold for plotting"),
):
    """
    Run the end-to-end pipeline for each BOLD run found in a BIDS dataset.
    """
    pairs = find_bids_runs(bids_root=bids, subject=subject, task=task)
    print(f"Found {len(pairs)} run(s).")
    for i, (func, events) in enumerate(pairs, 1):
        run_out = f"{out}/sub-{subject or 'ALL'}_run-{i}"
        print(f"[{i}/{len(pairs)}] func={func} events={events} -> out={run_out}")
        paths = run_pipeline(
            func=func,
            events=events,
            out_root=run_out,
            tr=tr,
            hrf=hrf,
            high_pass=high_pass,
            fwhm=fwhm,
            contrast=contrast,
            viz_threshold=viz_threshold,
        )
        for k, v in paths.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    app()
