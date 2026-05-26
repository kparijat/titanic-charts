from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_FILE = Path("train_and_test2.csv")
OUTPUT_DIR = Path("charts")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the CSV dataset and validate the columns required for charts."""
    df = pd.read_csv(path)

    required_columns = {"Pclass", "Age", "Fare", "2urvived"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and rename columns used by the charts."""
    cleaned = df.copy()

    cleaned = cleaned.rename(columns={"2urvived": "Survived"})
    cleaned["Survival Status"] = cleaned["Survived"].map(
        {0: "Did not survive", 1: "Survived"}
    )
    cleaned["Passenger Class"] = cleaned["Pclass"].map(
        {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}
    )

    cleaned = cleaned.dropna(subset=["Age", "Fare", "Pclass", "Survived"])

    return cleaned


def save_survival_rate_by_class(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a bar chart showing survival rate by passenger class."""
    survival_rate = df.groupby("Passenger Class")["Survived"].mean().reindex(
        ["1st Class", "2nd Class", "3rd Class"]
    )

    plt.figure(figsize=(8, 5))
    survival_rate.plot(kind="bar")
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "survival_rate_by_class.png", dpi=300)
    plt.close()


def save_age_distribution_by_survival(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a histogram comparing age distribution by survival status."""
    survived = df[df["Survived"] == 1]["Age"]
    not_survived = df[df["Survived"] == 0]["Age"]

    plt.figure(figsize=(8, 5))
    plt.hist([not_survived, survived], bins=20, label=["Did not survive", "Survived"])
    plt.title("Age Distribution by Survival Status")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "age_distribution_by_survival.png", dpi=300)
    plt.close()


def save_average_fare_by_class(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a bar chart showing average fare by passenger class."""
    average_fare = df.groupby("Passenger Class")["Fare"].mean().reindex(
        ["1st Class", "2nd Class", "3rd Class"]
    )

    plt.figure(figsize=(8, 5))
    average_fare.plot(kind="bar")
    plt.title("Average Fare by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Average Fare")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "average_fare_by_class.png", dpi=300)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw_data = load_dataset(DATA_FILE)
    data = prepare_dataset(raw_data)

    save_survival_rate_by_class(data, OUTPUT_DIR)
    save_age_distribution_by_survival(data, OUTPUT_DIR)
    save_average_fare_by_class(data, OUTPUT_DIR)

    print("Charts saved successfully:")
    print(f"- {OUTPUT_DIR / 'survival_rate_by_class.png'}")
    print(f"- {OUTPUT_DIR / 'age_distribution_by_survival.png'}")
    print(f"- {OUTPUT_DIR / 'average_fare_by_class.png'}")


if __name__ == "__main__":
    main()