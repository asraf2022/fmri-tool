from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional

def find_bids_runs(
    bids_root: str,
    subject: Optional[str] = None,   # e.g., "01" or "sub-01"
    task: Optional[str] = None,      # e.g., "faces"
) -> List[Tuple[str, str]]:
    """
    Return [(func_bold_path, events_tsv_path), ...] for a BIDS dataset.
    Looks for sub-*/func/*_bold.nii[.gz] and matching *_events.tsv.
    If task is given, require '*task-<task>*'.
    If subject is given, restrict to that subject only.

    Minimal helper; not full BIDS validation.
    """
    root = Path(bids_root)
    if not root.exists():
        raise FileNotFoundError(f"BIDS root not found: {root}")

    subj_glob = f"sub-{subject}" if subject and not subject.startswith("sub-") else (subject or "sub-*")
    func_dir_glob = root.glob(f"{subj_glob}/func")

    pairs: List[Tuple[str, str]] = []
    for func_dir in func_dir_glob:
        if not func_dir.is_dir():
            continue
        pattern = "*_bold.nii*"
        if task:
            pattern = f"*task-{task}*{pattern}"
        for bold in func_dir.glob(pattern):
            stem = bold.name.replace("_bold.nii.gz", "").replace("_bold.nii", "")
            ev = func_dir / f"{stem}_events.tsv"
            if ev.exists():
                pairs.append((str(bold), str(ev)))

    if not pairs:
        hint = f" (subject={subject})" if subject else ""
        hint += f" (task={task})" if task else ""
        raise FileNotFoundError(f"No BOLD/events pairs found in {root}{hint}.")
    return pairs
