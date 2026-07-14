from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path("/app/data/ab_test_data.csv")
SEED_BASE = 12345
CHANNELS = [
    {
        "channel": "organic",
        "size": 90000,
        "treatment_share": 0.50,
        "control_rate": 0.120,
        "treatment_rate": 0.118,
    },
    {
        "channel": "email",
        "size": 90000,
        "treatment_share": 0.50,
        "control_rate": 0.180,
        "treatment_rate": 0.178,
    },
    {
        "channel": "referral",
        "size": 90000,
        "treatment_share": 0.50,
        "control_rate": 0.090,
        "treatment_rate": 0.088,
    },
    {
        "channel": "paid_search",
        "size": 100000,
        "treatment_share": 0.40,
        "control_rate": 0.050,
        "treatment_rate": 0.048,
    },
]

OUTPUT_DIR = DATA_PATH.parent


def make_assignment_array(n, p, rng):
    n_treat = int(round(n * p))
    n_control = n - n_treat
    values = np.array(["treatment"] * n_treat + ["control"] * n_control)
    rng.shuffle(values)
    return values


def build_dataset(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    user_id = 1

    for config in CHANNELS:
        channel = config["channel"]
        n = config["size"]
        assignment = make_assignment_array(n, config["treatment_share"], rng)
        rates = np.where(assignment == "treatment", config["treatment_rate"], config["control_rate"])

        conversion = rng.random(size=n) < rates
        for assign, conv in zip(assignment, conversion.astype(int)):
            rows.append(
                {
                    "user_id": user_id,
                    "channel": channel,
                    "assignment": assign,
                    "converted": int(conv),
                }
            )
            user_id += 1

    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> dict:
    with pd.option_context("mode.chained_assignment", None):
        overall = df.groupby("assignment")["converted"].agg(["sum", "count"]).rename(columns={"sum": "conversions"})
        overall = overall.reindex(["control", "treatment"])
        metrics = {}
        for label in ["control", "treatment"]:
            conversions = int(overall.loc[label, "conversions"])
            users = int(overall.loc[label, "count"])
            metrics[label] = {
                "users": users,
                "conversions": conversions,
                "rate": conversions / users,
            }
    return metrics


def passes_paradox(df: pd.DataFrame) -> bool:
    metrics = summary(df)
    control_rate = metrics["control"]["rate"]
    treatment_rate = metrics["treatment"]["rate"]
    if treatment_rate <= control_rate:
        return False

    by_channel = []
    for channel, group in df.groupby("channel"):
        control = group[group["assignment"] == "control"]
        treatment = group[group["assignment"] == "treatment"]
        if treatment["converted"].mean() >= control["converted"].mean():
            return False
        by_channel.append(channel)

    return True


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for seed_offset in range(20):
        seed = SEED_BASE + seed_offset
        df = build_dataset(seed)
        if passes_paradox(df):
            df.to_csv(DATA_PATH, index=False)
            print(f"Generated dataset at {DATA_PATH} with seed {seed}")
            return

    raise RuntimeError("Unable to generate a dataset with the intended Simpson-style trap.")


if __name__ == "__main__":
    main()
