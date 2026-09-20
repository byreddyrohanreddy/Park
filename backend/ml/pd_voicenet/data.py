"""
data.py — Dataset manifest, subject/site parsing, GroupKFold helpers.

Shared utility for PD-VoiceNet. All other scripts import from here.
"""

import os
import re
import hashlib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import GroupKFold


# ---------------------------------------------------------------------------
# Site / subject parsing
# ---------------------------------------------------------------------------

def get_site(filename: str) -> str:
    """Extract canonical site code from filename using an explicit whitelist.
    
    Ensures that accidental prefix collisions (e.g., B1 vs a future B3) 
    don't silently misgroup datasets. Handles old anomalies like VA1/VA2 -> VA.
    """
    basename = os.path.basename(filename)
    
    # Explicit ordered prefixes. 
    # B1/B2/D1/D2 are distinct. PR/VA/VE/VI/VO/VU swallow their trailing numbers.
    valid_prefixes = ["B1", "B2", "D1", "D2", "AH", "FB", "PR", "VA", "VE", "VI", "VO", "VU"]
    
    for prefix in valid_prefixes:
        if basename.startswith(prefix):
            return prefix
            
    raise ValueError(f"Unknown site prefix in filename: {basename}")


def get_subject_id(filename: str) -> str:
    """Extract subject identifier from filename.
    
    HC pattern: SITE_SUBJECTID_UUID.wav
        e.g. AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav
        subject = AH_064F
        
    PD pattern: SITE_NUMERICID-UUID.wav
        e.g. AH_545616858-3A749CBC-3FEB-4D35-820E-E45C3E5B9B6A.wav
        subject = AH_545616858
        
    NOTE: On this dataset, every subject has exactly 1 recording.
    The SubjectAttentionPooling module handles N=1 correctly (trivial attention).
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split("_")
    
    if len(parts) >= 3:
        # HC format: SITE_SUBJECTID_UUID — take first two parts
        # Check if third part looks like a UUID (contains hyphens and is long)
        third = parts[2] if len(parts) > 2 else ""
        if len(third) > 8 and "-" in third:
            return f"{parts[0]}_{parts[1]}"
    
    if len(parts) >= 2:
        # PD format: SITE_NUMERICID-UUID — split second part on first hyphen
        second = parts[1]
        hyphen_idx = second.find("-")
        if hyphen_idx > 0:
            return f"{parts[0]}_{second[:hyphen_idx]}"
        return f"{parts[0]}_{second}"
    
    # Fallback: use full basename
    return basename


def get_recording_id(filename: str) -> str:
    """Return a unique, filesystem-safe recording ID.
    
    Uses the full basename (minus extension) to guarantee uniqueness.
    """
    return os.path.splitext(os.path.basename(filename))[0]


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------

def build_manifest(dataset_dir: str) -> pd.DataFrame:
    """Build a manifest DataFrame of all original recordings (excluding _world variants).
    
    Returns DataFrame with columns:
        recording_id, path, category (HC/PD), site, subject_id
    """
    records = []
    for category in ["HC", "PD"]:
        cat_dir = os.path.join(dataset_dir, category)
        if not os.path.exists(cat_dir):
            continue
        for f in sorted(os.listdir(cat_dir)):
            if not f.endswith(".wav"):
                continue
            if "_world" in f:
                continue
            path = os.path.join(cat_dir, f)
            records.append({
                "recording_id": get_recording_id(f),
                "path": path,
                "category": category,
                "site": get_site(f),
                "subject_id": get_subject_id(f),
            })
    
    df = pd.DataFrame(records)
    
    # Sanity checks
    assert len(df) > 0, f"No recordings found in {dataset_dir}"
    assert df["recording_id"].is_unique, "Duplicate recording IDs found"
    
    return df


def get_valid_sites(manifest: pd.DataFrame, min_count: int = 5) -> list:
    """Return sorted list of sites with at least min_count recordings."""
    counts = manifest["site"].value_counts()
    return sorted(counts[counts >= min_count].index.tolist())


def get_label(category: str) -> int:
    """HC=0, PD=1."""
    return 0 if category == "HC" else 1


# ---------------------------------------------------------------------------
# GroupKFold helpers
# ---------------------------------------------------------------------------

def subject_group_kfold_split(manifest: pd.DataFrame, n_splits: int = 5,
                               random_state: int = 42):
    """Yield (train_manifest, val_manifest) splits grouped by subject_id.
    
    Guarantees no subject appears on both sides of any split.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = manifest["subject_id"].values
    y = manifest["category"].apply(get_label).values
    X = np.arange(len(manifest))
    
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        yield manifest.iloc[train_idx].copy(), manifest.iloc[val_idx].copy()


def lodo_split(manifest: pd.DataFrame, held_out_site: str):
    """Split manifest into train (all sites except held_out) and test (held_out only).
    
    Returns (train_manifest, test_manifest).
    Asserts zero subject overlap.
    """
    test_mask = manifest["site"] == held_out_site
    train_df = manifest[~test_mask].copy()
    test_df = manifest[test_mask].copy()
    
    # Verify zero subject overlap
    train_subjects = set(train_df["subject_id"].unique())
    test_subjects = set(test_df["subject_id"].unique())
    overlap = train_subjects & test_subjects
    assert len(overlap) == 0, (
        f"Subject overlap between train and held-out site '{held_out_site}': {overlap}"
    )
    
    return train_df, test_df


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def hash_tensor(t) -> str:
    """Return a short hex digest of a tensor's bytes, for init verification."""
    return hashlib.md5(t.detach().cpu().numpy().tobytes()).hexdigest()[:12]


if __name__ == "__main__":
    # Quick self-test
    import sys
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "../../files/dataset"
    
    manifest = build_manifest(dataset_dir)
    print(f"Total recordings: {len(manifest)}")
    print(f"Categories: {dict(manifest['category'].value_counts())}")
    print(f"Sites: {dict(manifest['site'].value_counts())}")
    print(f"Unique subjects: {manifest['subject_id'].nunique()}")
    print(f"Valid sites (>=5): {get_valid_sites(manifest)}")
    
    # Check subject-recording ratio
    recs_per_subject = manifest.groupby("subject_id").size()
    print(f"Recordings per subject — min: {recs_per_subject.min()}, "
          f"max: {recs_per_subject.max()}, mean: {recs_per_subject.mean():.1f}")
