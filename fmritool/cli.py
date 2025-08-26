import typer
from pathlib import Path
from fmritool.preprocess import run_preprocess

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
    """
    Minimal fallback preprocessing: smooth the input NIfTI.
    """
    out_file = run_preprocess(func_path=func, out_dir=out, fwhm=fwhm)
    print(f"Saved preprocessed file: {out_file}")

@app.command("make-demo")
def make_demo(
    out_dir: str = typer.Option("examples", help="Where to place demo files"),
    tr: float = typer.Option(2.0, help="Repetition time (s) for info only"),
    n_vols: int = typer.Option(40, help="Number of timepoints (volumes)"),
):
    """
    Create a tiny synthetic 4D NIfTI + events.tsv so you can test the pipeline.
    """
    import numpy as np
    import nibabel as nib

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # make synthetic data: 64x64x10 voxels, n_vols timepoints
    shape = (64, 64, 10, n_vols)
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, size=shape).astype("float32")

    img = nib.Nifti1Image(data, affine=np.eye(4))
    func_path = out / "demo_func.nii.gz"
    nib.save(img, str(func_path))

    # simple events.tsv: four blocks alternating condA/condB (2 s duration)
    events_path = out / "events.tsv"
    events_tsv = (
        "onset\tduration\ttrial_type\n"
        "0\t2\tcondA\n"
        "20\t2\tcondB\n"
        "40\t2\tcondA\n"
        "60\t2\tcondB\n"
    )
    events_path.write_text(events_tsv)

    print(f"Demo functional: {func_path}")
    print(f"Demo events:     {events_path}")
    print(f"(TR info only: {tr}s)")

if __name__ == "__main__":
    app()
