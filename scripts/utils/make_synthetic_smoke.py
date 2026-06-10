#!/usr/bin/env python3
"""
Generate a fully SYNTHETIC smoke-test dataset (no real participant data).

This produces a 100-participant pickle that mirrors the column schema expected
by the analysis pipeline (steps 02-09), so the smoke test can run end-to-end
on any machine without access to any real cohort data.

All identifiers are fake (SUB_0001, SUB_0002, ...), the fMRI "data" arrays are
random Gaussian time series, and all clinical / demographic fields are filled
with randomized but schema-faithful values. Nothing here corresponds to a real
person.

Usage:
  python scripts/utils/make_synthetic_smoke.py \
      --out-pklz smoke_test_abide100.pkl \
      --n-per-group 50 \
      --seed 42
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd

N_ROI = 246
T_LEN = 120  # time points per synthetic scan (>= model crop length)

# Site labels are generic placeholders, not tied to any real acquisition site.
FAKE_SITES = [f"SITE_{i:02d}" for i in range(1, 13)]

# COMORBIDITY tokens kept generic; "ADHD" retained so step 05 ADHD logic exercises.
COMORBIDITY_CHOICES = [
    "none", "ADHD combined", "ADHD; anxiety", "anxiety disorder",
    "OCD", "specific phobia", np.nan,
]

HANDEDNESS = ["R", "L", "Ambi", "-9999"]


def rand_or_nan(rng, n, null_frac, lo, hi, integer=False):
    """Vector of length n in [lo,hi] with `null_frac` proportion set to NaN."""
    vals = rng.uniform(lo, hi, size=n)
    if integer:
        vals = np.round(vals)
    mask = rng.random(n) < null_frac
    vals = vals.astype(float)
    vals[mask] = np.nan
    return vals


def build_schema(rng, n):
    """Return a dict column -> array for the n synthetic rows."""
    cols = {}

    # ── Identifiers (all fake) ──────────────────────────────────────────────
    fake_ids = [f"SUB_{i:04d}" for i in range(1, n + 1)]
    cols["SUB_ID"] = fake_ids
    cols["subject_id"] = list(fake_ids)

    sites = rng.choice(FAKE_SITES, size=n)
    cols["SITE_ID"] = sites
    cols["SITE"] = sites
    cols["ABIDE"] = rng.choice([1, 2], size=n)

    # ── Diagnosis / group (1=ASD, 2=TDC, ABIDE convention) ──────────────────
    dx = np.array([1] * (n // 2) + [2] * (n - n // 2))
    rng.shuffle(dx)
    cols["DX_GROUP"] = dx
    cols["label"] = dx.astype(str)

    cols["COMORBIDITY"] = rng.choice(COMORBIDITY_CHOICES, size=n)
    cols["DSM_IV_TR"] = rand_or_nan(rng, n, 0.30, 0, 3, integer=True)
    cols["ASD_DSM_5"] = rand_or_nan(rng, n, 0.90, 0, 1, integer=True)

    # ── Demographics ────────────────────────────────────────────────────────
    cols["AGE_AT_SCAN"] = rng.uniform(7.0, 40.0, size=n)
    cols["SEX"] = rng.choice([1, 2], size=n)
    cols["HANDEDNESS_CATEGORY"] = rng.choice(HANDEDNESS, size=n)
    cols["EYE_STATUS_AT_SCAN"] = rng.choice([0, 1, 2], size=n).astype(float)
    cols["AGE_AT_MPRAGE"] = rand_or_nan(rng, n, 0.95, 8.0, 15.0)
    cols["BMI"] = rand_or_nan(rng, n, 0.85, 15.0, 30.0)

    # ── IQ ──────────────────────────────────────────────────────────────────
    cols["FIQ"] = rand_or_nan(rng, n, 0.06, 70, 140)
    cols["VIQ"] = rand_or_nan(rng, n, 0.20, 60, 140)
    cols["PIQ"] = rand_or_nan(rng, n, 0.17, 60, 145)
    for c in ["FIQ_TEST_TYPE", "VIQ_TEST_TYPE", "PIQ_TEST_TYPE"]:
        cols[c] = rng.choice(["WASI", "WISC", "DAS", np.nan], size=n)

    # ── ADI-R ───────────────────────────────────────────────────────────────
    cols["ADI_R_SOCIAL_TOTAL_A"] = rand_or_nan(rng, n, 0.74, 0, 30, integer=True)
    cols["ADI_R_VERBAL_TOTAL_BV"] = rand_or_nan(rng, n, 0.74, 0, 24, integer=True)
    cols["ADI_RRB_TOTAL_C"] = rand_or_nan(rng, n, 0.74, 0, 12, integer=True)
    cols["ADI_R_ONSET_TOTAL_D"] = rand_or_nan(rng, n, 0.78, 0, 5, integer=True)
    cols["ADI_R_RSRCH_RELIABLE"] = rand_or_nan(rng, n, 0.74, 0, 1, integer=True)

    # ── ADOS ────────────────────────────────────────────────────────────────
    socaff = rand_or_nan(rng, n, 0.33, 0, 14, integer=True)
    cols["ADOS_GOTHAM_SOCAFFECT"] = pd.Series(socaff).apply(
        lambda v: str(int(v)) if pd.notna(v) else np.nan
    ).values
    cols["ADOS_GOTHAM_RRB"] = rand_or_nan(rng, n, 0.86, 0, 4, integer=True)
    cols["ADOS_GOTHAM_TOTAL"] = rand_or_nan(rng, n, 0.70, 0, 16, integer=True)
    cols["ADOS_GOTHAM_SEVERITY"] = rand_or_nan(rng, n, 0.70, 1, 9, integer=True)
    cols["ADOS_2_SOCAFFECT"] = rand_or_nan(rng, n, 0.83, 5, 16, integer=True)
    cols["ADOS_2_RRB"] = rand_or_nan(rng, n, 0.83, 1, 6, integer=True)
    cols["ADOS_2_TOTAL"] = rand_or_nan(rng, n, 0.83, 7, 22, integer=True)
    cols["ADOS_2_SEVERITY_TOTAL"] = rand_or_nan(rng, n, 0.83, 3, 10, integer=True)

    # ── Other social/screen scales ──────────────────────────────────────────
    cols["SRS_VERSION"] = rand_or_nan(rng, n, 0.51, 1, 2, integer=True)
    cols["SRS_RAW_TOTAL"] = rand_or_nan(rng, n, 0.40, 0, 148)
    cols["SCQ_TOTAL"] = rand_or_nan(rng, n, 0.81, 0, 37, integer=True)
    cols["AQ_TOTAL"] = rand_or_nan(rng, n, 0.94, 17, 45, integer=True)

    # ── RBS-R subscales ─────────────────────────────────────────────────────
    rbsr6 = {
        "RBSR_6SUBSCALE_STEREOTYPED": 7, "RBSR_6SUBSCALE_SELF-INJURIOUS": 7,
        "RBSR_6SUBSCALE_COMPULSIVE": 6, "RBSR_6SUBSCALE_RITUALISTIC": 9,
        "RBSR_6SUBSCALE_SAMENESS": 16, "RBSR_6SUBSCALE_RESTRICTED": 6,
        "RBSR_6SUBSCALE_TOTAL": 41,
    }
    for c, hi in rbsr6.items():
        cols[c] = rand_or_nan(rng, n, 0.82, 0, hi, integer=True)
    rbsr5 = {
        "RBSR_5SUBSCALE_STEREOTYPIC": 6, "RBSR_5SUBSCALE_SELF-INJURIOUS": 3,
        "RBSR_5SUBSCALE_COMPULSIVE": 3, "RBSR_5SUBSCALE_RITUALISTIC": 8,
        "RBSR_5SUBSCALE_RESTRICTED": 5, "RBSR_5SUBSCALE_TOTAL": 23,
    }
    for c, hi in rbsr5.items():
        cols[c] = rand_or_nan(rng, n, 0.91, 0, hi, integer=True)

    # ── BRIEF executive-function scales ─────────────────────────────────────
    cols["BRIEF_VERSION"] = rand_or_nan(rng, n, 0.77, 1, 1, integer=True)
    cols["BRIEF_INFORMANT"] = rand_or_nan(rng, n, 0.82, 1, 8, integer=True)
    brief_t = [
        "BRIEF_INHIBIT_T", "BRIEF_SHIFT_T", "BRIEF_EMOTIONAL_T", "BRIEF_BRI_T",
        "BRIEF_INITIATE_T", "BRIEF_WORKING_T", "BRIEF_PLAN_T",
        "BRIEF_ORGANIZATION_T", "BRIEF_MONITOR_T", "BRIEF_MI_T", "BRIEF_GEC_T",
    ]
    for c in brief_t:
        cols[c] = rand_or_nan(rng, n, 0.77, 30, 90)
    cols["BRIEF_INCONSISTENCY_SCORE"] = rand_or_nan(rng, n, 0.78, 0, 7, integer=True)
    cols["BRIEF_NEGATIVITY_SCORE"] = rand_or_nan(rng, n, 0.78, 0, 7, integer=True)

    # ── CBCL 6-18 ───────────────────────────────────────────────────────────
    cbcl618 = [
        "CBCL_6-18_ACTIVITIES_T", "CBCL_6-18_SOCIAL_T", "CBCL_6-18_SCHOOL_T",
        "CBCL_6-18_TOTAL_COMPETENCE_T", "CBCL_6-18_ANXIOUS_T",
        "CBCL_6-18_WITHDRAWN_T", "CBCL_6-18_SOMATIC_COMPAINT_T",
        "CBCL_6-18_SOCIAL_PROBLEM_T", "CBCL_6-18_THOUGHT_T",
        "CBCL_6-18_ATTENTION_T", "CBCL_6-18_RULE_T", "CBCL_6-18_AGGRESSIVE_T",
        "CBCL_6-18_INTERNAL_T", "CBCL_6-18_EXTERNAL_T",
        "CBCL_6-18_TOTAL_PROBLEM_T", "CBCL_6-18_AFFECTIVE_T",
        "CBCL_6-18_ANXIETY_T", "CBCL_6-18_SOMATIC_PROBLEM_T",
        "CBCL_6-18_ATTENTION_DEFICIT_T", "CBCL_6-18_OPPOSITIONAL_T",
        "CBCL_6-18_CONDUCT_T", "CBCL_6-18_SLUGGISH_T", "CBCL_6-18_OBSESSIVE_T",
        "CBCL_6-18_POST_TRAUMATIC_T",
    ]
    for c in cbcl618:
        cols[c] = rand_or_nan(rng, n, 0.85, 50, 80)

    # ── CBCL 1.5-5 (all-NaN in reference sample; preserve as empty column) ──
    cbcl15 = [
        "CBCL_1.5-5_EMOTION_T", "CBCL_1.5-5_ANXIOUS_T", "CBCL_1.5-5_SOMANTIC_T",
        "CBCL_1.5-5_WITHDRAWN_T", "CBCL_1.5-5_SLEEP_T",
        "CBCL_1.5-5_ATTENTION_PROBLEM_T", "CBCL_1.5-5_AGGRESSIVE_T",
        "CBCL_1.5-5_INTERNAL_T", "CBCL_1.5-5_EXTERNAL_T", "CBCL_1.5-5_TOTAL_T",
        "CBCL_1.5-5_STRESS_T", "CBCL_1.5-5_AFFECTIVE_T", "CBCL_1.5-5_ANXIETY_T",
        "CBCL_1.5-5_PERVASIVE_T", "CBCL_1.5-5_ATTENTION_DEFICIT_T",
        "CBCL_1.5-5_OPPOSITIONAL_T",
    ]
    for c in cbcl15:
        cols[c] = np.full(n, np.nan)

    # ── fMRI data: random Gaussian time series, QC fields within thresholds ─
    cols["data"] = [rng.standard_normal((T_LEN, N_ROI)).astype(np.float32)
                    for _ in range(n)]
    cols["tr"] = rng.uniform(0.5, 3.0, size=n)
    cols["mean_fd"] = rng.uniform(0.03, 0.35, size=n)            # all <= 0.5
    cols["percentofvolsrepaired"] = rng.uniform(0.0, 9.5, size=n)  # all <= 10

    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-pklz", default="smoke_test_abide100.pkl")
    ap.add_argument("--n-per-group", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n = args.n_per_group * 2
    rng = np.random.default_rng(args.seed)
    cols = build_schema(rng, n)

    # Column order matching the expected schema
    order = [
        "SITE_ID", "ABIDE", "SITE", "SUB_ID", "DX_GROUP", "COMORBIDITY",
        "DSM_IV_TR", "ASD_DSM_5", "AGE_AT_SCAN", "SEX", "HANDEDNESS_CATEGORY",
        "FIQ", "VIQ", "PIQ", "FIQ_TEST_TYPE", "VIQ_TEST_TYPE", "PIQ_TEST_TYPE",
        "ADI_R_SOCIAL_TOTAL_A", "ADI_R_VERBAL_TOTAL_BV", "ADI_RRB_TOTAL_C",
        "ADI_R_ONSET_TOTAL_D", "ADI_R_RSRCH_RELIABLE", "ADOS_GOTHAM_SOCAFFECT",
        "ADOS_GOTHAM_RRB", "ADOS_GOTHAM_TOTAL", "ADOS_GOTHAM_SEVERITY",
        "ADOS_2_SOCAFFECT", "ADOS_2_RRB", "ADOS_2_TOTAL", "ADOS_2_SEVERITY_TOTAL",
        "SRS_VERSION", "SRS_RAW_TOTAL", "SCQ_TOTAL", "AQ_TOTAL",
        "EYE_STATUS_AT_SCAN", "AGE_AT_MPRAGE", "BMI",
    ]
    order += [c for c in cols if c not in order and c not in
              ("data", "subject_id", "label", "tr", "mean_fd",
               "percentofvolsrepaired")]
    order += ["subject_id", "data", "label", "tr", "mean_fd",
              "percentofvolsrepaired"]

    df = pd.DataFrame({c: cols[c] for c in order})

    with open(args.out_pklz, "wb") as f:
        pickle.dump(df, f)

    size_mb = os.path.getsize(args.out_pklz) / 1e6
    print(f"Synthetic smoke-test dataset written: {args.out_pklz} ({size_mb:.1f} MB)")
    print(f"  Participants : {len(df)}  ({(df.DX_GROUP==1).sum()} ASD / "
          f"{(df.DX_GROUP==2).sum()} TDC)")
    print(f"  Columns      : {len(df.columns)}")
    print(f"  IDs          : all synthetic (SUB_0001 ... SUB_{len(df):04d})")
    print( "  NOTE         : fully synthetic; contains no real participant data.")


if __name__ == "__main__":
    main()
