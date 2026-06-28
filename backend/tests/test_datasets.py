from backend.app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())


def _upload_csv(filename: str, csv_content: bytes) -> dict:
    response = client.post(
        "/api/datasets/upload",
        files={"file": (filename, csv_content, "text/csv")},
    )

    assert response.status_code == 201

    return response.json()["dataset"]


def test_upload_csv_returns_dataset_detail() -> None:
    # 測試 CSV 上傳後，API 是否回傳 dataset metadata。
    csv_content = b"name,age,city\nAlice,30,Taipei\nBob,25,Taichung\n"

    dataset = _upload_csv("sample.csv", csv_content)

    assert dataset["original_filename"] == "sample.csv"
    assert dataset["row_count"] == 2
    assert dataset["column_count"] == 3
    assert dataset["columns"] == ["name", "age", "city"]
    assert dataset["status"] == "ready"
    assert len(dataset["preview_rows"]) == 2


def test_upload_rejects_non_csv_file() -> None:
    # 測試非 CSV 檔案會被擋下，避免錯誤資料進入 pipeline。
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only CSV files are supported in Phase 1."


def test_list_datasets_returns_uploaded_items() -> None:
    # 測試 dataset list API 是否能回傳已上傳資料。
    csv_content = b"name,age\nAlice,30\nBob,25\n"
    _upload_csv("list_sample.csv", csv_content)

    response = client.get("/api/datasets")

    assert response.status_code == 200

    payload = response.json()
    assert "datasets" in payload
    assert "total" in payload
    assert payload["total"] >= 1


def test_dataset_preview_returns_rows_and_columns() -> None:
    # 先上傳一份資料，再用回傳的 dataset_id 測試 preview API。
    csv_content = b"product,price,quantity\nBook,300,2\nPen,20,10\n"

    dataset = _upload_csv("sales.csv", csv_content)
    dataset_id = dataset["id"]

    preview_response = client.get(f"/api/datasets/{dataset_id}/preview?max_rows=1")

    assert preview_response.status_code == 200

    payload = preview_response.json()
    assert payload["dataset_id"] == dataset_id
    assert payload["columns"] == ["product", "price", "quantity"]
    assert payload["row_count"] == 2
    assert payload["preview_row_count"] == 1
    assert payload["rows"][0]["product"] == "Book"
