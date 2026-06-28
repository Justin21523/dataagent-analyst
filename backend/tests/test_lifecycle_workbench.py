from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _upload_lifecycle_dataset() -> str:
    csv_content = (
        b"customer_id,age,signup_date,monthly_charges,tenure_months,contract_type,churn\n"
        b"C0001,29,2024-01-01,799,12,Monthly,No\n"
        b"C0002,41,2024-01-02,1299,4,Monthly,Yes\n"
        b"C0003,,2024-01-03,999,24,One Year,No\n"
        b"C0004,52,2024-01-04,1599,2,Monthly,Yes\n"
        b"C0005,46,2024-01-05,899,36,Two Year,No\n"
        b"C0006,31,2024-01-06,1099,7,Monthly,Yes\n"
        b"C0007,27,2024-01-07,699,18,One Year,No\n"
        b"C0008,63,2024-01-08,1899,3,Monthly,Yes\n"
        b"C0009,38,2024-01-09,1199,15,One Year,No\n"
        b"C0010,44,2024-01-10,1399,6,Monthly,Yes\n"
        b"C0011,33,2024-01-11,850,16,One Year,No\n"
        b"C0012,58,2024-01-12,1750,5,Monthly,Yes\n"
    )
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("lifecycle.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 201

    return response.json()["dataset"]["id"]


def _train_model(dataset_id: str) -> str:
    response = client.post(
        f"/api/ml/{dataset_id}/train",
        json={
            "target_column": "churn",
            "selected_models": ["logistic_regression"],
            "test_size": 0.25,
            "random_state": 42,
        },
    )

    assert response.status_code == 201

    return response.json()["best_model_id"]


def test_dataset_versions_and_transform_pipeline() -> None:
    dataset_id = _upload_lifecycle_dataset()

    versions_response = client.get(f"/api/datasets/{dataset_id}/versions")
    assert versions_response.status_code == 200
    assert versions_response.json()["latest_version_id"] == "v1"

    transform_payload = {
        "fill_missing": [
            {
                "column": "age",
                "strategy": "median",
            }
        ],
        "drop_columns": ["customer_id"],
        "datetime_parts": [
            {
                "column": "signup_date",
                "parts": ["year", "month"],
            }
        ],
    }

    preview_response = client.post(
        f"/api/datasets/{dataset_id}/transform/preview",
        json=transform_payload,
    )
    assert preview_response.status_code == 200
    assert preview_response.json()["source_version_id"] == "v1"
    assert "signup_date_year" in preview_response.json()["columns"]

    apply_response = client.post(
        f"/api/datasets/{dataset_id}/transform",
        json=transform_payload,
    )
    assert apply_response.status_code == 201

    payload = apply_response.json()
    assert payload["version"]["version_id"] == "v2"
    assert payload["profile_diff"]["column_delta"] == 1


def test_model_lifecycle_and_diagnostics() -> None:
    dataset_id = _upload_lifecycle_dataset()
    model_id = _train_model(dataset_id)

    status_response = client.patch(
        f"/api/ml/models/{model_id}/status",
        json={"lifecycle_status": "production"},
    )
    assert status_response.status_code == 200
    assert status_response.json()["lifecycle_status"] == "production"

    production_response = client.get(f"/api/ml/{dataset_id}/models/production")
    assert production_response.status_code == 200
    assert production_response.json()["id"] == model_id

    threshold_response = client.post(
        f"/api/ml/models/{model_id}/threshold-analysis",
        json={"positive_class": "Yes", "thresholds": [0.4, 0.5]},
    )
    assert threshold_response.status_code == 200
    assert len(threshold_response.json()["points"]) == 2

    segment_response = client.post(
        f"/api/ml/models/{model_id}/segment-metrics",
        json={"segment_column": "contract_type"},
    )
    assert segment_response.status_code == 200
    assert segment_response.json()["segments"]

    what_if_response = client.post(
        f"/api/ml/models/{model_id}/what-if",
        json={
            "base_record": {
                "age": 40,
                "monthly_charges": 1100,
                "tenure_months": 9,
                "contract_type": "Monthly",
            },
            "scenarios": [
                {
                    "name": "higher_tenure",
                    "changes": {
                        "tenure_months": 24,
                    },
                }
            ],
        },
    )
    assert what_if_response.status_code == 200
    assert what_if_response.json()["results"][0]["scenario"] == "higher_tenure"

    migration_response = client.post(f"/api/ml/models/{model_id}/migration-check")
    assert migration_response.status_code == 200
    assert "checks" in migration_response.json()


def test_drift_report_and_bundle_round_trip() -> None:
    dataset_id = _upload_lifecycle_dataset()
    model_id = _train_model(dataset_id)
    transform_response = client.post(
        f"/api/datasets/{dataset_id}/transform",
        json={
            "fill_missing": [
                {
                    "column": "age",
                    "strategy": "constant",
                    "value": 99,
                }
            ],
            "iqr_clip": [
                {
                    "column": "monthly_charges",
                    "factor": 1.5,
                }
            ],
        },
    )
    assert transform_response.status_code == 201

    drift_response = client.post(
        "/api/drift/reports",
        json={
            "dataset_id": dataset_id,
            "reference_version_id": "v1",
            "current_version_id": "v2",
            "model_id": model_id,
            "target_column": "churn",
        },
    )
    assert drift_response.status_code == 200
    drift_payload = drift_response.json()
    assert drift_payload["report_id"]
    assert drift_payload["feature_drift"]

    export_response = client.post(
        "/api/bundles/export",
        json={"dataset_id": dataset_id},
    )
    assert export_response.status_code == 200
    bundle_path = export_response.json()["bundle_path"]

    with open(bundle_path, "rb") as bundle_file:
        import_response = client.post(
            "/api/bundles/import",
            files={"file": ("bundle.zip", bundle_file.read(), "application/zip")},
        )

    assert import_response.status_code == 200
    assert import_response.json()["imported_dataset_id"] != dataset_id


def test_drift_recommendation_and_challenger_retraining_flow() -> None:
    dataset_id = _upload_lifecycle_dataset()
    champion_id = _train_model(dataset_id)

    promote_response = client.patch(
        f"/api/ml/models/{champion_id}/status",
        json={"lifecycle_status": "production"},
    )
    assert promote_response.status_code == 200

    transform_response = client.post(
        f"/api/datasets/{dataset_id}/transform",
        json={
            "fill_missing": [
                {
                    "column": "age",
                    "strategy": "constant",
                    "value": 72,
                }
            ],
            "drop_duplicate_rows": True,
        },
    )
    assert transform_response.status_code == 201
    current_version_id = transform_response.json()["version"]["version_id"]

    drift_response = client.post(
        "/api/drift/reports",
        json={
            "dataset_id": dataset_id,
            "reference_version_id": "v1",
            "current_version_id": current_version_id,
            "model_id": champion_id,
            "target_column": "churn",
        },
    )
    assert drift_response.status_code == 200
    assert "score" in drift_response.json()["retraining_recommendation"]

    plan_response = client.post(
        f"/api/ml/models/{champion_id}/retrain-plan",
        json={
            "current_version_id": current_version_id,
            "reference_version_id": "v1",
            "drift_report_id": drift_response.json()["report_id"],
        },
    )
    assert plan_response.status_code == 200
    assert plan_response.json()["selected_model"] == "logistic_regression"

    retrain_response = client.post(
        f"/api/ml/models/{champion_id}/retrain",
        json={
            "current_version_id": current_version_id,
            "reference_version_id": "v1",
            "drift_report_id": drift_response.json()["report_id"],
        },
    )
    assert retrain_response.status_code == 200
    challenger_id = retrain_response.json()["challenger_model"]["id"]
    assert retrain_response.json()["challenger_model"]["lifecycle_status"] == "candidate"
    assert retrain_response.json()["comparison"]["models"]

    promote_challenger_response = client.post(
        f"/api/ml/models/{champion_id}/promote-challenger",
        json={"challenger_model_id": challenger_id},
    )
    assert promote_challenger_response.status_code == 200
    assert promote_challenger_response.json()["promoted_model"]["id"] == challenger_id
    assert promote_challenger_response.json()["promoted_model"]["lifecycle_status"] == "production"
