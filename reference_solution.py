import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest

DATA_PATH = Path("/app/data/ab_test_data.csv")
OUTPUT_PATH = Path("/app/output/eda_report.json")


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def compute_overall(df: pd.DataFrame) -> dict:
    control = df[df["assignment"] == "control"]
    treatment = df[df["assignment"] == "treatment"]

    control_users = len(control)
    treatment_users = len(treatment)
    control_conversions = int(control["converted"].sum())
    treatment_conversions = int(treatment["converted"].sum())
    control_conversion_rate = control_conversions / control_users
    treatment_conversion_rate = treatment_conversions / treatment_users
    relative_difference = (treatment_conversion_rate - control_conversion_rate) / control_conversion_rate

    _, p_value = proportions_ztest(
        count=np.array([treatment_conversions, control_conversions]),
        nobs=np.array([treatment_users, control_users]),
        alternative="two-sided",
    )

    se = np.sqrt(
        treatment_conversion_rate * (1 - treatment_conversion_rate) / treatment_users
        + control_conversion_rate * (1 - control_conversion_rate) / control_users
    )
    ci_lower = relative_difference - 1.96 * se
    ci_upper = relative_difference + 1.96 * se

    return {
        "control_users": control_users,
        "treatment_users": treatment_users,
        "control_conversions": control_conversions,
        "treatment_conversions": treatment_conversions,
        "control_conversion_rate": control_conversion_rate,
        "treatment_conversion_rate": treatment_conversion_rate,
        "relative_difference": relative_difference,
        "p_value": float(p_value),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def compute_channel_stats(df: pd.DataFrame) -> list:
    rows = []
    for channel, group in df.groupby("channel"):
        control = group[group["assignment"] == "control"]
        treatment = group[group["assignment"] == "treatment"]
        control_users = len(control)
        treatment_users = len(treatment)
        control_rate = int(control["converted"].sum()) / control_users
        treatment_rate = int(treatment["converted"].sum()) / treatment_users
        rows.append(
            {
                "channel": channel,
                "control_users": control_users,
                "treatment_users": treatment_users,
                "control_conversion_rate": control_rate,
                "treatment_conversion_rate": treatment_rate,
                "relative_difference": (treatment_rate - control_rate) / control_rate,
                "assignment_ratio": {
                    "control": control_users / len(group),
                    "treatment": treatment_users / len(group),
                },
            }
        )
    return rows


def compute_adjusted(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["treatment_flag"] = (df["assignment"] == "treatment").astype(int)
    df["original_channel"] = df["channel"]
    df = pd.get_dummies(df, columns=["channel"], drop_first=True)
    X_cols = [col for col in df.columns if col.startswith("channel_") and col != "original_channel"]
    X = sm.add_constant(df[["treatment_flag"] + X_cols]).astype(float)
    y = df["converted"].astype(float)

    model = sm.Logit(y, X).fit(disp=False, maxiter=200)
    X_control = X.copy()
    X_control["treatment_flag"] = 0
    X_treatment = X.copy()
    X_treatment["treatment_flag"] = 1
    estimate = (model.predict(X_treatment) - model.predict(X_control)).mean()

    boot_estimates = []
    rng = np.random.default_rng(1234)
    for _ in range(200):
        sample = df.sample(frac=1.0, replace=True, random_state=int(rng.integers(1_000_000)))
        Xs = sm.add_constant(sample[["treatment_flag"] + X_cols]).astype(float)
        ys = sample["converted"].astype(float)
        try:
            boot = sm.Logit(ys, Xs).fit(disp=False, maxiter=200)
        except Exception:
            continue
        Xc = Xs.copy()
        Xt = Xs.copy()
        Xc["treatment_flag"] = 0
        Xt["treatment_flag"] = 1
        boot_estimates.append((boot.predict(Xt) - boot.predict(Xc)).mean())

    if len(boot_estimates) < 100:
        raise RuntimeError("Bootstrap did not generate enough replicates.")

    ci_lower = float(np.percentile(boot_estimates, 2.5))
    ci_upper = float(np.percentile(boot_estimates, 97.5))
    p_value = float(2 * min(np.mean(np.array(boot_estimates) >= 0), np.mean(np.array(boot_estimates) <= 0)))
    significant = (ci_lower > 0) or (ci_upper < 0)

    assignment_ratios = df.groupby("original_channel")["treatment_flag"].mean()
    irregular_channel = assignment_ratios.sub(0.5).abs().idxmax()

    return {
        "method": "logistic_regression_adjustment",
        "estimate": float(estimate),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": bool(significant),
        "irregular_channel": irregular_channel,
    }


def build_report() -> dict:
    df = load_data()
    return {
        "dataset_path": str(DATA_PATH),
        "overall": compute_overall(df),
        "channels": compute_channel_stats(df),
        "adjusted": compute_adjusted(df),
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
