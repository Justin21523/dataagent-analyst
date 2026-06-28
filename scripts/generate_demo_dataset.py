import csv
import random
from datetime import date, timedelta
from pathlib import Path

OUTPUT_PATH = Path("data/samples/customer_churn_demo.csv")
ROW_COUNT = 300
RANDOM_SEED = 42


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def build_customer(index: int) -> dict[str, object]:
    age = random.randint(19, 72)
    tenure_months = random.randint(1, 60)
    contract_type = random.choices(
        ["Monthly", "One Year", "Two Year"],
        weights=[0.55, 0.28, 0.17],
        k=1,
    )[0]

    payment_method = random.choice(
        [
            "Credit Card",
            "Bank Transfer",
            "Electronic Wallet",
        ]
    )

    gender = random.choice(["Female", "Male"])
    region = random.choice(["North", "Central", "South", "East"])
    support_tickets = random.choices(
        [0, 1, 2, 3, 4, 5],
        weights=[0.25, 0.28, 0.2, 0.14, 0.08, 0.05],
        k=1,
    )[0]

    satisfaction_score = random.randint(1, 5)
    is_active = random.random() > 0.22

    contract_adjustment = {
        "Monthly": 260,
        "One Year": 120,
        "Two Year": 0,
    }[contract_type]

    monthly_charges = 520 + contract_adjustment + support_tickets * 55 + random.randint(-110, 620)

    monthly_charges = max(399, monthly_charges)

    # churn probability 刻意與多個 feature 相關，方便 ML Demo 學到訊號。
    churn_probability = 0.12
    churn_probability += 0.22 if contract_type == "Monthly" else 0
    churn_probability += support_tickets * 0.055
    churn_probability += 0.12 if satisfaction_score <= 2 else 0
    churn_probability += 0.1 if not is_active else 0
    churn_probability += 0.08 if monthly_charges >= 1300 else 0
    churn_probability -= min(tenure_months, 36) * 0.006

    churn_probability = clamp(churn_probability, 0.04, 0.88)
    churn = "Yes" if random.random() < churn_probability else "No"

    signup_date = date(2021, 1, 1) + timedelta(
        days=random.randint(0, 1500),
    )

    notes = random.choice(
        [
            "Regular customer",
            "Asked about pricing",
            "Uses mobile app",
            "Contacted support",
            "Requested discount",
            "",
        ]
    )

    # 少量 missing values 讓 EDA Dashboard 有可展示的資料品質問題。
    age_value: object = age if random.random() > 0.03 else ""
    charges_value: object = monthly_charges if random.random() > 0.02 else ""
    satisfaction_value: object = satisfaction_score if random.random() > 0.04 else ""

    return {
        "customer_id": f"C{index:05d}",
        "age": age_value,
        "gender": gender,
        "region": region,
        "signup_date": signup_date.isoformat(),
        "monthly_charges": charges_value,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "is_active": str(is_active).lower(),
        "support_tickets": support_tickets,
        "satisfaction_score": satisfaction_value,
        "notes": notes,
        "churn": churn,
    }


def main() -> None:
    random.seed(RANDOM_SEED)

    rows = [build_customer(index) for index in range(1, ROW_COUNT + 1)]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
