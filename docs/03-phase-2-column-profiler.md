# Phase 2: Dataset Preview & Column Profiler

## Goal

Add schema intelligence to uploaded CSV datasets.

This phase allows the system to infer column types, detect missing values, count unique values, identify target candidates, and generate a dataset schema summary.

## Deliverables

- Column profiler service
- Column type inference
- Missing value analysis
- Unique value analysis
- Sample value extraction
- Numeric statistics
- Top value distribution
- Dataset schema summary API
- Column profiles API
- Frontend schema overview
- Frontend column profile cards

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/datasets/{dataset_id}/columns` | Get column profiles |
| GET | `/api/datasets/{dataset_id}/schema` | Get schema summary and column profiles |

## Supported Inferred Types

| Type | Description |
|---|---|
| `numeric` | Numeric values or values convertible to numbers |
| `categorical` | Low-cardinality text values |
| `datetime` | Values convertible to datetime |
| `boolean` | Boolean-like values such as true/false, yes/no, 0/1 |
| `text` | Long free-text values |
| `unknown` | Empty or unsupported columns |

## Semantic Roles

| Role | Description |
|---|---|
| `identifier` | ID-like columns |
| `target_candidate` | Potential ML target columns |
| `datetime` | Date or time columns |
| `feature_candidate` | General feature columns |

## Data Flow

```text
Dataset ID
→ Load CSV with DatasetService
→ Infer column types
→ Analyze missing values
→ Count unique values
→ Extract sample values
→ Generate numeric / datetime stats
→ Build schema summary
→ Return API response
