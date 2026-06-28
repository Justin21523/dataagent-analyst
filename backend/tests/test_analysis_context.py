from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _upload_analysis_sample_dataset() -> str:
    # 測試資料包含 missing、duplicate、numeric、categorical 與 target。
    csv_content = (
        b"customer_id,age,monthly_charges,tenure_months,contract_type,churn\n"
        b"C0001,29,799,12,Monthly,No\n"
        b"C0002,41,1299,4,Monthly,Yes\n"
        b"C0003,35,999,24,One Year,No\n"
        b"C0004,52,1599,2,Monthly,Yes\n"
        b"C0005,46,899,36,Two Year,No\n"
        b"C0006,31,1099,7,Monthly,Yes\n"
        b"C0007,27,699,18,One Year,No\n"
        b"C0008,63,1899,3,Monthly,Yes\n"
        b"C0009,38,1199,15,One Year,No\n"
        b"C0010,44,1399,6,Monthly,No\n"
        b"C0011,,850,16,One Year,No\n"
        b"C0012,58,1750,5,Monthly,No\n"
        b"C0012,58,1750,5,Monthly,No\n"
    )

    response = client.post(
        "/api/datasets/upload",
        files={
            "file": (
                "analysis_context_sample.csv",
                csv_content,
                "text/csv",
            )
        },
    )

    assert response.status_code == 201

    return response.json()["dataset"]["id"]


def test_analysis_context_returns_canonical_structure() -> None:
    dataset_id = _upload_analysis_sample_dataset()

    response = client.post(
        f"/api/analysis/{dataset_id}/context",
        json={
            "target_column": "churn",
            "user_goal": "Find churn drivers and recommend actions.",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["analysis_context_version"] == "1.0"
    assert payload["dataset_id"] == dataset_id
    assert payload["user_goal"] == ("Find churn drivers and recommend actions.")

    assert payload["data_quality"]["overall_score"] >= 0
    assert len(payload["data_quality"]["dimensions"]) == 4

    dimension_keys = {dimension["key"] for dimension in payload["data_quality"]["dimensions"]}

    assert dimension_keys == {
        "completeness",
        "consistency",
        "type_stability",
        "outlier_stability",
    }

    assert payload["target_analysis"]["target_column"] == "churn"
    assert payload["target_analysis"]["task_type"] == "classification"
    assert payload["target_analysis"]["class_distribution"]

    assert "findings" in payload
    assert "recommended_next_actions" in payload
    assert payload["rag_context"] == []


def test_numeric_unique_features_are_not_identifiers() -> None:
    dataset_id = _upload_analysis_sample_dataset()

    response = client.post(
        f"/api/analysis/{dataset_id}/context",
        json={"target_column": "churn"},
    )

    assert response.status_code == 200

    columns = {column["name"]: column for column in response.json()["column_profiles"]}

    assert columns["customer_id"]["semantic_role"] == "identifier"
    assert columns["age"]["semantic_role"] == "feature_candidate"
    assert columns["monthly_charges"]["semantic_role"] == "feature_candidate"
    assert columns["tenure_months"]["semantic_role"] == "feature_candidate"


def test_analysis_context_auto_detects_target_candidate() -> None:
    dataset_id = _upload_analysis_sample_dataset()

    response = client.post(
        f"/api/analysis/{dataset_id}/context",
        json={"user_goal": "Prepare a supervised learning analysis."},
    )

    assert response.status_code == 200
    assert response.json()["target_analysis"]["target_column"] == "churn"


def test_analysis_context_rejects_unknown_target() -> None:
    dataset_id = _upload_analysis_sample_dataset()

    response = client.post(
        f"/api/analysis/{dataset_id}/context",
        json={"target_column": "not_existing_target"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Target column not found: not_existing_target"
