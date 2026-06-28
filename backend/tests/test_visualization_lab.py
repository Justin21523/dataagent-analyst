from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _upload_visualization_lab_dataset() -> str:
    csv_content = (
        b"customer_id,signup_date,age,monthly_charges,"
        b"tenure_months,contract_type,region,churn\n"
        b"C0001,2024-01-01,20,700,2,Monthly,North,No\n"
        b"C0002,2024-01-02,25,800,3,Monthly,North,No\n"
        b"C0003,2024-01-03,,900,4,One Year,South,Yes\n"
        b"C0004,2024-01-04,60,1900,40,Monthly,South,Yes\n"
        b"C0005,2024-01-05,42,1000,12,Two Year,Central,No\n"
        b"C0006,2024-01-06,31,1200,6,Monthly,North,Yes\n"
        b"C0007,2024-01-07,28,650,20,One Year,East,No\n"
        b"C0008,2024-01-08,55,1750,3,Monthly,South,Yes\n"
        b"C0009,2024-01-09,38,1100,15,One Year,Central,No\n"
        b"C0010,2024-01-10,44,1450,5,Monthly,North,Yes\n"
        b"C0011,2024-01-11,33,,16,One Year,East,No\n"
        b"C0012,2024-01-12,58,1800,4,Monthly,South,Yes\n"
    )

    response = client.post(
        "/api/datasets/upload",
        files={
            "file": (
                "visualization_lab_sample.csv",
                csv_content,
                "text/csv",
            )
        },
    )

    assert response.status_code == 201

    return response.json()["dataset"]["id"]


def test_visualization_lab_returns_analysis_charts() -> None:
    dataset_id = _upload_visualization_lab_dataset()

    response = client.post(
        f"/api/visualizations/{dataset_id}/lab",
        json={
            "target_column": "churn",
            "sample_rows": 80,
            "max_numeric_columns": 8,
            "max_categories": 12,
        },
    )

    assert response.status_code == 200

    payload = response.json()
    chart_types = {chart["chart_type"] for chart in payload["charts"]}

    assert payload["dataset_id"] == dataset_id
    assert payload["summary"]["target_column"] == "churn"
    assert payload["summary"]["chart_count"] == len(payload["charts"])

    assert "missing_bar" in chart_types
    assert "missing_heatmap" in chart_types
    assert "correlation_heatmap" in chart_types
    assert "boxplot" in chart_types
    assert "target_distribution" in chart_types

    assert payload["summary"]["numeric_columns"]
    assert payload["summary"]["categorical_columns"]


def test_correlation_heatmap_contains_matrix_coordinates() -> None:
    dataset_id = _upload_visualization_lab_dataset()

    response = client.post(
        f"/api/visualizations/{dataset_id}/lab",
        json={"target_column": "churn"},
    )

    assert response.status_code == 200

    correlation_chart = next(
        chart for chart in response.json()["charts"] if chart["chart_type"] == "correlation_heatmap"
    )

    assert correlation_chart["data"]["x_labels"]
    assert correlation_chart["data"]["y_labels"]
    assert correlation_chart["data"]["values"]

    first_value = correlation_chart["data"]["values"][0]

    assert len(first_value) == 3


def test_custom_scatter_chart_is_generated() -> None:
    dataset_id = _upload_visualization_lab_dataset()

    response = client.post(
        f"/api/visualizations/{dataset_id}/build",
        json={
            "chart_type": "scatter",
            "x_column": "age",
            "y_column": "monthly_charges",
            "target_column": "churn",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["chart_type"] == "scatter"
    assert payload["columns"] == [
        "age",
        "monthly_charges",
    ]
    assert payload["data"]["source"]


def test_custom_scatter_rejects_non_numeric_column() -> None:
    dataset_id = _upload_visualization_lab_dataset()

    response = client.post(
        f"/api/visualizations/{dataset_id}/build",
        json={
            "chart_type": "scatter",
            "x_column": "contract_type",
            "y_column": "monthly_charges",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Column must be numeric: contract_type"
