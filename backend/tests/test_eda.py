from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _upload_eda_sample_dataset() -> str:
    # 測試資料刻意包含缺失值、重複列、數值欄位與明顯 outlier。
    csv_content = (
        b"customer_id,age,monthly_charges,tenure_months,churn\n"
        b"C0001,20,100,2,No\n"
        b"C0002,25,200,3,No\n"
        b"C0003,,300,4,Yes\n"
        b"C0003,,300,4,Yes\n"
        b"C0004,60,5000,40,Yes\n"
    )

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("eda_sample.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 201

    return response.json()["dataset"]["id"]


def test_eda_summary_returns_core_sections() -> None:
    dataset_id = _upload_eda_sample_dataset()

    response = client.get(f"/api/eda/{dataset_id}/summary")

    assert response.status_code == 200

    payload = response.json()

    assert payload["dataset_id"] == dataset_id
    assert payload["row_count"] == 5
    assert payload["column_count"] == 5
    assert 0 <= payload["data_quality_score"] <= 100
    assert payload["data_quality_grade"] in {
        "Excellent",
        "Good",
        "Fair",
        "Needs Attention",
    }
    assert payload["missing"]["total_missing_cells"] == 2
    assert payload["duplicates"]["duplicate_row_count"] == 1
    assert len(payload["numeric_statistics"]["columns"]) >= 2
    assert len(payload["outliers"]["columns"]) >= 1
    assert "recommendations" in payload


def test_missing_analysis_returns_missing_columns() -> None:
    dataset_id = _upload_eda_sample_dataset()

    response = client.get(f"/api/eda/{dataset_id}/missing")

    assert response.status_code == 200

    payload = response.json()
    columns = {column["name"]: column for column in payload["columns"]}

    assert payload["total_missing_cells"] == 2
    assert columns["age"]["missing_count"] == 2
    assert columns["age"]["missing_ratio"] == 0.4


def test_numeric_statistics_returns_numeric_columns() -> None:
    dataset_id = _upload_eda_sample_dataset()

    response = client.get(f"/api/eda/{dataset_id}/statistics")

    assert response.status_code == 200

    payload = response.json()
    columns = {column["name"]: column for column in payload["columns"]}

    assert "age" in columns
    assert "monthly_charges" in columns
    assert columns["monthly_charges"]["max"] == 5000.0


def test_outlier_analysis_detects_iqr_outlier() -> None:
    dataset_id = _upload_eda_sample_dataset()

    response = client.get(f"/api/eda/{dataset_id}/outliers")

    assert response.status_code == 200

    payload = response.json()
    columns = {column["name"]: column for column in payload["columns"]}

    assert "monthly_charges" in columns
    assert columns["monthly_charges"]["method"] == "iqr"
    assert columns["monthly_charges"]["outlier_count"] >= 1


def test_correlation_analysis_returns_matrix() -> None:
    dataset_id = _upload_eda_sample_dataset()

    response = client.get(f"/api/eda/{dataset_id}/correlation")

    assert response.status_code == 200

    payload = response.json()

    assert payload["method"] == "pearson"
    assert "age" in payload["columns"]
    assert "monthly_charges" in payload["columns"]
    assert isinstance(payload["matrix"], dict)
    assert "strongest_pairs" in payload
