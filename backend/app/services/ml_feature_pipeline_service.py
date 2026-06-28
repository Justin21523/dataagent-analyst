from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from backend.app.ml.transformers import (
    DateTimeFeatureExtractor,
    TextSeriesCleaner,
)
from backend.app.schemas.analysis_schema import (
    AnalysisColumnSummary,
)
from backend.app.schemas.ml_workbench_schema import (
    MLFeatureGroups,
    MLPipelineStep,
    MLWorkbenchPlanRequest,
)


class MLFeaturePipelineError(Exception):
    """Raised when an ML feature pipeline cannot be built."""


@dataclass
class MLFeaturePipelineBundle:
    preprocessor: ColumnTransformer
    feature_groups: MLFeatureGroups
    preprocessing_steps: list[MLPipelineStep]
    estimated_feature_count: int
    warnings: list[str]


class MLFeaturePipelineService:
    def build_bundle(
        self,
        dataframe: pd.DataFrame,
        profiles: list[AnalysisColumnSummary],
        request: MLWorkbenchPlanRequest,
        target_column: str | None,
    ) -> MLFeaturePipelineBundle:
        profile_map = {profile.name: profile for profile in profiles}

        selected_columns, excluded_columns = self._select_columns(
            dataframe=dataframe,
            profiles=profiles,
            request=request,
            target_column=target_column,
        )

        numeric_columns = []
        categorical_columns = []
        datetime_columns = []
        text_columns = []
        warnings = []

        for column in selected_columns:
            profile = profile_map[column]

            if profile.inferred_type == "numeric":
                numeric_columns.append(column)
            elif profile.inferred_type in {
                "categorical",
                "boolean",
            }:
                if request.include_text and self._looks_like_text_feature(
                    column_name=column,
                    series=dataframe[column],
                ):
                    text_columns.append(column)
                else:
                    categorical_columns.append(column)
            elif profile.inferred_type == "datetime":
                if request.include_datetime:
                    datetime_columns.append(column)
                else:
                    excluded_columns.append(column)
            elif profile.inferred_type == "text":
                if request.include_text:
                    text_columns.append(column)
                else:
                    excluded_columns.append(column)
            else:
                excluded_columns.append(column)

        if not any(
            [
                numeric_columns,
                categorical_columns,
                datetime_columns,
                text_columns,
            ]
        ):
            raise MLFeaturePipelineError("No valid feature columns are available.")

        if text_columns:
            warnings.append(
                "Text features use dense TF-IDF output in this portfolio "
                "version; monitor memory usage on large datasets."
            )

        high_cardinality = [
            column for column in categorical_columns if profile_map[column].cardinality == "high"
        ]

        if high_cardinality:
            warnings.append(
                "High-cardinality categorical features may create many "
                "encoded columns: " + ", ".join(high_cardinality)
            )

        estimated_feature_count = (
            len(numeric_columns)
            + len(datetime_columns) * 6
            + sum(
                max(
                    1,
                    profile_map[column].unique_count,
                )
                for column in categorical_columns
            )
            + len(text_columns) * request.text_max_features
        )

        estimated_cells = int(dataframe.shape[0]) * estimated_feature_count

        if estimated_feature_count > 5000:
            warnings.append("Estimated transformed feature count exceeds 5,000.")

        if estimated_cells > 8_000_000:
            raise MLFeaturePipelineError(
                "Estimated dense feature matrix is too large. "
                "Reduce categories, text features, or selected columns."
            )

        transformers: list[tuple[str, Any, Any]] = []
        preprocessing_steps = []

        if numeric_columns:
            numeric_steps = [
                (
                    "imputer",
                    SimpleImputer(
                        strategy=request.numeric_imputer,
                    ),
                )
            ]

            scaler = self._build_scaler(request.scaler)

            operations = [
                f"{request.numeric_imputer}_imputer",
            ]

            if scaler is not None:
                numeric_steps.append(
                    (
                        "scaler",
                        scaler,
                    )
                )
                operations.append(f"{request.scaler}_scaler")

            transformers.append(
                (
                    "numeric",
                    Pipeline(steps=numeric_steps),
                    numeric_columns,
                )
            )

            preprocessing_steps.append(
                MLPipelineStep(
                    id="numeric",
                    label="Numeric Pipeline",
                    columns=numeric_columns,
                    operations=operations,
                )
            )

        if categorical_columns:
            minimum_frequency = (
                request.one_hot_min_frequency if request.one_hot_min_frequency > 1 else None
            )

            encoder = OneHotEncoder(
                handle_unknown=("infrequent_if_exist" if minimum_frequency else "ignore"),
                min_frequency=minimum_frequency,
                sparse_output=False,
            )

            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="most_frequent",
                                ),
                            ),
                            (
                                "encoder",
                                encoder,
                            ),
                        ]
                    ),
                    categorical_columns,
                )
            )

            preprocessing_steps.append(
                MLPipelineStep(
                    id="categorical",
                    label="Categorical Pipeline",
                    columns=categorical_columns,
                    operations=[
                        "most_frequent_imputer",
                        "one_hot_encoder",
                    ],
                )
            )

        if datetime_columns:
            transformers.append(
                (
                    "datetime",
                    Pipeline(
                        steps=[
                            (
                                "extractor",
                                DateTimeFeatureExtractor(),
                            ),
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="most_frequent",
                                ),
                            ),
                            (
                                "scaler",
                                StandardScaler(),
                            ),
                        ]
                    ),
                    datetime_columns,
                )
            )

            preprocessing_steps.append(
                MLPipelineStep(
                    id="datetime",
                    label="Datetime Pipeline",
                    columns=datetime_columns,
                    operations=[
                        "year",
                        "month",
                        "day",
                        "dayofweek",
                        "quarter",
                        "is_month_end",
                        "most_frequent_imputer",
                        "standard_scaler",
                    ],
                )
            )

        for text_column in text_columns:
            transformer_name = "text_" + self._slugify(text_column)

            transformers.append(
                (
                    transformer_name,
                    Pipeline(
                        steps=[
                            (
                                "cleaner",
                                TextSeriesCleaner(),
                            ),
                            (
                                "tfidf",
                                TfidfVectorizer(
                                    max_features=(request.text_max_features),
                                    ngram_range=(1, 2),
                                    min_df=1,
                                ),
                            ),
                        ]
                    ),
                    text_column,
                )
            )

            preprocessing_steps.append(
                MLPipelineStep(
                    id=transformer_name,
                    label=f"Text Pipeline: {text_column}",
                    columns=[text_column],
                    operations=[
                        "fill_empty_text",
                        "tfidf_vectorizer",
                        "unigram_and_bigram",
                    ],
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=True,
        )

        return MLFeaturePipelineBundle(
            preprocessor=preprocessor,
            feature_groups=MLFeatureGroups(
                numeric=numeric_columns,
                categorical=categorical_columns,
                datetime=datetime_columns,
                text=text_columns,
                excluded=sorted(set(excluded_columns)),
            ),
            preprocessing_steps=preprocessing_steps,
            estimated_feature_count=estimated_feature_count,
            warnings=warnings,
        )

    def _select_columns(
        self,
        dataframe: pd.DataFrame,
        profiles: list[AnalysisColumnSummary],
        request: MLWorkbenchPlanRequest,
        target_column: str | None,
    ) -> tuple[list[str], list[str]]:
        profile_map = {profile.name: profile for profile in profiles}

        requested_features = list(dict.fromkeys(request.feature_columns))

        requested_exclusions = set(request.excluded_columns)

        unknown_requested = [
            column
            for column in (requested_features + list(requested_exclusions))
            if column not in dataframe.columns
        ]

        if unknown_requested:
            raise MLFeaturePipelineError(
                "Unknown feature columns: " + ", ".join(sorted(set(unknown_requested)))
            )

        if requested_features:
            candidates = requested_features
        else:
            candidates = [profile.name for profile in profiles if profile.name in dataframe.columns]

        selected = []
        excluded = []

        for column in candidates:
            profile = profile_map[column]

            should_exclude = (
                column == target_column
                or column in requested_exclusions
                or profile.semantic_role == "identifier"
                or (profile.semantic_role == "target_candidate" and column != target_column)
                or profile.inferred_type == "unknown"
            )

            if should_exclude:
                excluded.append(column)
                continue

            selected.append(column)

        for profile in profiles:
            if profile.name not in selected and profile.name not in excluded:
                excluded.append(profile.name)

        return selected, excluded

    def _build_scaler(
        self,
        scaler_name: str,
    ):
        if scaler_name == "standard":
            return StandardScaler()

        if scaler_name == "robust":
            return RobustScaler()

        if scaler_name == "minmax":
            return MinMaxScaler()

        if scaler_name == "none":
            return None

        raise MLFeaturePipelineError(f"Unsupported scaler: {scaler_name}")

    def _looks_like_text_feature(
        self,
        column_name: str,
        series: pd.Series,
    ) -> bool:
        normalized_name = column_name.strip().lower()

        if normalized_name in {
            "comment",
            "comments",
            "description",
            "message",
            "note",
            "notes",
            "review",
            "summary",
            "text",
        }:
            return True

        non_null_series = series.dropna()

        if non_null_series.empty:
            return False

        average_length = float(non_null_series.astype(str).str.len().mean())

        return average_length >= 20

    def _slugify(
        self,
        value: str,
    ) -> str:
        return "".join(
            character if character.isalnum() else "_" for character in value.lower()
        ).strip("_")
