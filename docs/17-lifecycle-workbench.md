# Lifecycle Workbench

## Goal

Turn DataAgent Analyst into a reproducible tabular ML lifecycle workbench: data versions, transformation recipes, model lifecycle, drift monitoring, challenger retraining, and portable artifacts.

## Data Versioning

Every uploaded dataset starts with `v1`.

Derived versions are created through transformation recipes and stored as CSV artifacts. The version registry records:

- dataset ID
- version ID
- source version ID
- version index
- row and column counts
- columns
- content hash
- transformation recipe
- profile diff
- warnings

Existing APIs default to the latest dataset version when `version_id` is omitted.

## Transformation Recipes

Supported transformation operations:

- drop columns
- rename columns
- cast numeric, datetime, string, boolean, and category columns
- fill missing values with mean, median, mode, or constant
- drop rows with missing values
- drop duplicate rows
- IQR outlier clipping
- derive datetime parts

Preview endpoints do not persist artifacts. Apply endpoints create a derived version.

## Model Lifecycle

Model lifecycle statuses:

- `candidate`: newly trained model
- `staging`: reviewed but not production
- `production`: active champion model
- `archived`: previous champion or retired model

Only one model per dataset, target, and task type should be production. Promoting a new production model archives the previous production model in that scope.

## Drift Center

Drift reports compare a reference dataset version against a current dataset version.

Current drift checks:

- schema drift: added columns, removed columns, type changes
- numeric feature drift: PSI and KS statistic
- categorical feature drift: PSI and Jensen-Shannon distance
- target drift when labels exist
- prediction drift when a model is supplied
- performance drift when labels and model predictions exist

PSI interpretation:

- `< 0.10`: stable
- `0.10 - 0.25`: warning
- `>= 0.25`: drift

Each report includes a retraining recommendation with a score, action, and reasons.

## Champion / Challenger Retraining

Retraining starts from a supervised champion model.

The challenger reuses:

- target column
- task type
- feature columns
- model family
- stored training config when available

The challenger trains on a current dataset version and starts as `candidate`.

Comparison metric:

- classification: `f1_macro`, higher is better
- regression: `rmse`, lower is better

The API returns a recommendation:

- `promote`: challenger ranks ahead of champion
- `review`: comparison is inconclusive
- `keep_champion`: champion still ranks ahead

Promotion is explicit by default. Auto-promote exists as an API option but should remain disabled in the UI unless the workflow is intentionally automated.

## Bundle Migration

Bundle export creates a zip file containing:

- manifest
- dataset metadata
- dataset versions
- model records and artifacts
- report records and Markdown files
- drift report records

Bundle import creates new IDs to avoid collisions and imports artifacts into local runtime storage.

## Deferred Work

PostgreSQL persistence, Qdrant-backed RAG, and dynamic agent planning are intentionally deferred until the lifecycle contracts stabilize.
