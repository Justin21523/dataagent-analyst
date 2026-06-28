# Phase 1: CSV Upload & Dataset Registry

## Goal

Build the first real data entry point for DataAgent Analyst.

This phase allows users to upload CSV files, store raw files, create dataset metadata, list uploaded datasets, and preview dataset rows.

## Deliverables

- CSV upload API
- Dataset registry stored as JSON
- Dataset list API
- Dataset detail API
- Dataset preview API
- Frontend upload panel
- Frontend dataset list
- Frontend preview table
- Dataset API tests
- Sample CSV dataset

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/datasets/upload` | Upload a CSV file |
| GET | `/api/datasets` | List uploaded datasets |
| GET | `/api/datasets/{dataset_id}` | Get dataset metadata |
| GET | `/api/datasets/{dataset_id}/preview` | Preview dataset rows |

## Data Flow

```text
CSV Upload
→ Validate file extension and size
→ Save raw CSV file
→ Parse CSV with pandas
→ Detect encoding
→ Create dataset metadata
→ Store metadata in JSON registry
→ Return dataset detail
