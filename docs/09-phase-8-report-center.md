# Phase 8: Report Center

## Goal

Add a Markdown-based analysis report center to DataAgent Analyst.

This phase generates deterministic reports from dataset metadata, schema summary, EDA results, visualization recommendations, and ML model results.

## Backend Deliverables

- Report schema models
- Report service
- Report API routes
- Markdown report generator
- Report registry
- Report file storage
- Report tests

## Frontend Deliverables

- Report Center section
- Generate report button
- Report history list
- Markdown report viewer
- Markdown download button

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/reports/{dataset_id}/generate` | Generate Markdown report |
| GET | `/api/reports/dataset/{dataset_id}` | List reports by dataset |
| GET | `/api/reports/{report_id}` | Get report detail |

## Report Flow

```text
Dataset ID
→ Load dataset metadata
→ Load schema summary
→ Load EDA summary
→ Load visualization recommendations
→ Load trained model results
→ Build Markdown report
→ Save report file
→ Save report registry record
→ Render report in frontend
git add .
