# Phase 4: Visualization Recommendation Engine

## Goal

Add automatic chart recommendation and chart rendering to DataAgent Analyst.

This phase recommends charts based on column profiles, EDA signals, and dataset structure. The backend returns Chart.js-compatible chart data, and the frontend renders charts in the Visualization Lab.

## Backend Deliverables

- Visualization schema models
- Visualization service
- Visualization API route
- Missing values chart recommendation
- Numeric histogram recommendation
- Categorical bar chart recommendation
- Datetime trend line recommendation
- Numeric scatter plot recommendation
- Target summary chart recommendation

## Frontend Deliverables

- Chart.js integration
- Visualization Lab section
- Chart recommendation cards
- Auto-rendered chart grid
- Chart renderer module
- Visualization view module

## API Endpoint

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/visualizations/{dataset_id}/recommendations` | Get chart recommendations and Chart.js data |

## Recommendation Logic

```text
Dataset ID
→ Load CSV
→ Profile columns
→ Detect missing values
→ Recommend data quality chart
→ Recommend numeric distributions
→ Recommend categorical distributions
→ Recommend datetime trends
→ Recommend numeric relationships
→ Recommend target-based summaries
→ Return Chart.js-compatible data
