# Phase 3: EDA Core Engine

## Goal

Build the core exploratory data analysis engine for DataAgent Analyst.

This phase adds missing value analysis, duplicate row analysis, numeric statistics, outlier detection, correlation analysis, data quality scoring, and frontend EDA dashboard rendering.

## Backend Deliverables

- EDA schema models
- EDA service
- EDA API routes
- Missing value analysis
- Duplicate analysis
- Numeric statistics
- IQR outlier detection
- Pearson correlation matrix
- Data quality score
- Rule-based EDA recommendations

## Frontend Deliverables

- Modular JavaScript structure
- Modular CSS structure
- EDA quality summary
- Missing value table
- Numeric statistics table
- Outlier analysis table
- Correlation pair table
- EDA recommendation cards

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/eda/{dataset_id}/summary` | Full EDA summary |
| GET | `/api/eda/{dataset_id}/missing` | Missing value analysis |
| GET | `/api/eda/{dataset_id}/duplicates` | Duplicate row analysis |
| GET | `/api/eda/{dataset_id}/statistics` | Numeric statistics |
| GET | `/api/eda/{dataset_id}/outliers` | IQR outlier analysis |
| GET | `/api/eda/{dataset_id}/correlation` | Pearson correlation analysis |

## EDA Data Flow

```text
Dataset ID
→ Load CSV through DatasetService
→ Reuse ColumnProfilerService
→ Run missing value analysis
→ Run duplicate analysis
→ Calculate numeric statistics
→ Detect IQR outliers
→ Calculate Pearson correlation
→ Calculate data quality score
→ Return EDA summary
