from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _upload_visualization_sample_dataset() -> str:
    # 測試資料包含數值、類別、日期、缺失值與 target，讓推薦引擎能產生多種圖表。
    csv_content = (
        b"customer_id,signup_date,age,monthly_charges,tenure_months,contract_type,churn\n"
        b"C0001,2024-01-01,20,100,2,Monthly,No\n"
        b"C0002,2024-01-02,25,200,3,Monthly,No\n"
        b"C0003,2024-01-03,,300,4,One Year,Yes\n"
        b"C0004,2024-01-04,60,5000,40,Monthly,Yes\n"
        b"C0005,2024-01-05,42,800,12,Two Year,No\n"
    )

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("visualization_sample.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 201

    return response.json()["dataset"]["id"]


def test_visualization_recommendations_returns_chart_list() -> None:
    dataset_id = _upload_visualization_sample_dataset()

    response = client.get(f"/api/visualizations/{dataset_id}/recommendations")

    assert response.status_code == 200

    payload = response.json()

    assert payload["dataset_id"] == dataset_id
    assert payload["total"] >= 4
    assert len(payload["recommendations"]) == payload["total"]


def test_visualization_recommendations_include_expected_chart_families() -> None:
    dataset_id = _upload_visualization_sample_dataset()

    response = client.get(f"/api/visualizations/{dataset_id}/recommendations")

    assert response.status_code == 200

    recommendations = response.json()["recommendations"]
    chart_families = {chart["chart_family"] for chart in recommendations}

    assert "data_quality" in chart_families
    assert "distribution" in chart_families
    assert "category_distribution" in chart_families
    assert "time_trend" in chart_families
    assert "relationship" in chart_families


def test_visualization_chart_data_is_chartjs_compatible() -> None:
    dataset_id = _upload_visualization_sample_dataset()

    response = client.get(f"/api/visualizations/{dataset_id}/recommendations")

    assert response.status_code == 200

    first_chart = response.json()["recommendations"][0]

    assert "labels" in first_chart["data"]
    assert "datasets" in first_chart["data"]
    assert isinstance(first_chart["data"]["datasets"], list)
    assert "label" in first_chart["data"]["datasets"][0]
    assert "data" in first_chart["data"]["datasets"][0]
