# fmritool

A tiny, beginner-friendly fMRI analysis tool (demo) that:
- smooths a functional NIfTI,
- fits a first-level GLM with Nilearn,
- outputs a Z-map.

We'll grow this into a full pipeline (FSL/SPM via Nipype, group analysis, QC reports, Streamlit UI).


## Demo Output

After running the demo + first-level GLM, you’ll get a Z-map and preview images:

<p align="center">
  <img src="first_level/zmap_condA-condB_glass.png" alt="Glass brain" width="45%">
  &nbsp;&nbsp;
  <img src="first_level/zmap_condA-condB_stat.png" alt="Stat map" width="45%">
</p>

**How to reproduce:**
```bash
fmritool make-demo
fmritool prep --func examples/demo_func.nii.gz --out derivatives
fmritool first-level --func derivatives/preproc_demo_func.nii.gz --events examples/events.tsv --out first_level
fmritool viz --zmap first_level/zmap_condA-condB.nii.gz --threshold 2.0


## Demo Output

## Demo Output

![Glass brain](https://raw.githubusercontent.com/asraf2022/fmri-tool/main/first_level/zmap_condA-condB_glass.png)
![Stat map](https://raw.githubusercontent.com/asraf2022/fmri-tool/main/first_level/zmap_condA-condB_stat.png)
