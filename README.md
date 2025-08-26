# fmritool

A tiny, beginner-friendly fMRI analysis tool (demo) that:
- smooths a functional NIfTI,
- fits a first-level GLM with Nilearn,
- outputs a Z-map.

We'll grow this into a full pipeline (FSL/SPM via Nipype, group analysis, QC reports, Streamlit UI).



**How to reproduce:**
```bash
fmritool make-demo
fmritool prep --func examples/demo_func.nii.gz --out derivatives
fmritool first-level --func derivatives/preproc_demo_func.nii.gz --events examples/events.tsv --out first_level
fmritool viz --zmap first_level/zmap_condA-condB.nii.gz --threshold 2.0
```