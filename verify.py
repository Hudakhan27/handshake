import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest

DATA_PATH = Path("/app/data/ab_test_data.csv")
REPORT_PATH = Path("/app/output/eda_report.json")


def load_report() -> dict:
    with REPORT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected dataset at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def compute_overall_statistics(df: pd.DataFrame) -> dict:
    control = df[df["assignment"] == "control"]
    treatment = df[df["assignment"] == "treatment"]

    control_users = len(control)
    treatment_users = len(treatment)
    control_conversions = int(control["converted"].sum())
    treatment_conversions = int(treatment["converted"].sum())
    control_rate = control_conversions / control_users
    treatment_rate = treatment_conversions / treatment_users
    relative_difference = (treatment_rate - control_rate) / control_rate

    stat, p_value = proportions_ztest(
        count=np.array([treatment_conversions, control_conversions]),
        nobs=np.array([treatment_users, control_users]),
        alternative="two-sided",
    )

    se = np.sqrt(
        treatment_rate * (1 - treatment_rate) / treatment_users
        + control_rate * (1 - control_rate) / control_users
    )
    ci_lower = relative_difference - 1.96 * se
    ci_upper = relative_difference + 1.96 * se

    return {
        "control_users": control_users,
        "treatment_users": treatment_users,
        "control_conversions": control_conversions,
        "treatment_conversions": treatment_conversions,
        "control_conversion_rate": control_rate,
        "treatment_conversion_rate": treatment_rate,
        "relative_difference": relative_difference,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def compute_channel_statistics(df: pd.DataFrame) -> dict:
    result = {}
    for channel, group in df.groupby("channel"):
        control = group[group["assignment"] == "control"]
        treatment = group[group["assignment"] == "treatment"]
        control_users = len(control)
        treatment_users = len(treatment)
        result[channel] = {
            "channel": channel,
            "control_users": control_users,
            "treatment_users": treatment_users,
            "control_conversion_rate": int(control["converted"].sum()) / control_users,
            "treatment_conversion_rate": int(treatment["converted"].sum()) / treatment_users,
            "relative_difference": (
                int(treatment["converted"].sum()) / treatment_users
                - int(control["converted"].sum()) / control_users
            )
            / (int(control["converted"].sum()) / control_users),
            "assignment_ratio": {
                "control": control_users / len(group),
                "treatment": treatment_users / len(group),
            },
        }
    return result


def compute_adjusted_statistics(df: pd.DataFrame) -> dict:
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
    for i in range(200):
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
        raise RuntimeError("Bootstrap failed to generate enough replicates.")

    ci_lower = np.percentile(boot_estimates, 2.5)
    ci_upper = np.percentile(boot_estimates, 97.5)
    p_value = 2 * min(
        np.mean(np.array(boot_estimates) >= 0),
        np.mean(np.array(boot_estimates) <= 0),
    )
    significant = (ci_lower > 0) or (ci_upper < 0)

    channel_counts = df[[col for col in df.columns if col.startswith("channel_")]].copy()
    original_channels = [col.replace("channel_", "") for col in channel_counts.columns]
    assignment_ratios = df.groupby("original_channel")["treatment_flag"].mean()
    irregular_channel = assignment_ratios.sub(0.5).abs().idxmax()

    return {
        "method": "logistic_regression_adjustment",
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": bool(significant),
        "irregular_channel": irregular_channel,
    }


def compare_values(name: str, actual, expected, tolerance=1e-8) -> bool:
    if isinstance(actual, float) or isinstance(expected, float):
        return abs(actual - expected) <= tolerance
    return actual == expected


def validate_report(report: dict, expected: dict, channel_stats: dict, adjusted: dict) -> None:
    errors = []

    if report.get("dataset_path") != str(DATA_PATH):
        errors.append(f"dataset_path mismatch: expected {DATA_PATH}, got {report.get('dataset_path')}")

    for key, expected_value in expected["overall"].items():
        actual = report.get("overall", {}).get(key)
        if key in {"control_conversion_rate", "treatment_conversion_rate", "relative_difference", "p_value", "ci_lower", "ci_upper"}:
            if not compare_values(key, float(actual), float(expected_value), tolerance=1e-6):
                errors.append(f"overall.{key}: expected {expected_value}, got {actual}")
        else:
            if actual != expected_value:
                errors.append(f"overall.{key}: expected {expected_value}, got {actual}")

    if not isinstance(report.get("channels"), list) or len(report["channels"]) != len(channel_stats):
        errors.append("channels array is missing or has incorrect length")
    else:
        for channel_obj in report["channels"]:
            channel = channel_obj.get("channel")
            if channel not in channel_stats:
                errors.append(f"unexpected channel entry: {channel}")
                continue
            expected_channel = channel_stats[channel]
            for field in ["control_users", "treatment_users"]:
                if channel_obj.get(field) != expected_channel[field]:
                    errors.append(f"channels[{channel}].{field}: expected {expected_channel[field]}, got {channel_obj.get(field)}")
            for field in ["control_conversion_rate", "treatment_conversion_rate", "relative_difference"]:
                if not compare_values(field, float(channel_obj.get(field)), float(expected_channel[field]), tolerance=1e-6):
                    errors.append(f"channels[{channel}].{field}: expected {expected_channel[field]}, got {channel_obj.get(field)}")
            assignment_ratio = channel_obj.get("assignment_ratio", {})
            if not compare_values("assignment_ratio.control", float(assignment_ratio.get("control")), expected_channel["assignment_ratio"]["control"], tolerance=1e-6):
                errors.append(f"channels[{channel}].assignment_ratio.control: expected {expected_channel['assignment_ratio']['control']}, got {assignment_ratio.get('control')}")
            if not compare_values("assignment_ratio.treatment", float(assignment_ratio.get("treatment")), expected_channel["assignment_ratio"]["treatment"], tolerance=1e-6):
                errors.append(f"channels[{channel}].assignment_ratio.treatment: expected {expected_channel['assignment_ratio']['treatment']}, got {assignment_ratio.get('treatment')}")

    for key, expected_value in adjusted.items():
        actual = report.get("adjusted", {}).get(key)
        if key in {"estimate", "ci_lower", "ci_upper", "p_value"}:
            if not compare_values(key, float(actual), float(expected_value), tolerance=1e-6):
                errors.append(f"adjusted.{key}: expected {expected_value}, got {actual}")
        else:
            if actual != expected_value:
                errors.append(f"adjusted.{key}: expected {expected_value}, got {actual}")

    if errors:
        raise AssertionError("Report validation failed:\n" + "\n".join(errors))


def main() -> None:
    report = load_report()
    df = load_data()
    expected_overall = compute_overall_statistics(df)
    channel_stats = compute_channel_statistics(df)
    adjusted_stats = compute_adjusted_statistics(df)

    expected = {"overall": expected_overall}
    validate_report(report, expected, channel_stats, adjusted_stats)
    print("PASS: /app/output/eda_report.json matches the expected values.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
