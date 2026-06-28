import math
from collections import defaultdict
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline

from backend.app.core.config import Settings
from backend.app.schemas.explainability_schema import (
    ExplainabilityImportanceItem,
    ShapBeeswarmPoint,
    ShapLocalContribution,
    ShapLocalExplanation,
    ShapSummary,
)


class ShapExplainabilityService:
    TREE_MODEL_NAMES = {
        "RandomForestClassifier",
        "RandomForestRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
    }

    LINEAR_MODEL_NAMES = {
        "LogisticRegression",
        "Ridge",
        "Lasso",
        "LinearSVC",
        "LinearSVR",
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def explain(
        self,
        pipeline: Pipeline,
        x_holdout: pd.DataFrame,
        local_row_position: int,
        sample_size: int,
        background_size: int,
        random_state: int,
        positive_class: str | None,
    ) -> ShapSummary:
        try:
            import shap
        except ImportError:
            return ShapSummary(
                available=False,
                warning=("SHAP is not installed. Install requirements.txt again."),
            )

        if x_holdout.empty:
            return ShapSummary(
                available=False,
                warning="Holdout data is empty.",
            )

        preprocessor = pipeline.named_steps.get("preprocessor")
        estimator = pipeline.named_steps.get("model")

        if preprocessor is None or estimator is None:
            return ShapSummary(
                available=False,
                warning=("Model pipeline does not contain preprocessor and model steps."),
            )

        estimator_name = estimator.__class__.__name__

        if estimator_name in self.TREE_MODEL_NAMES:
            explainer_type = "TreeExplainer"
        elif estimator_name in self.LINEAR_MODEL_NAMES:
            explainer_type = "LinearExplainer"
        else:
            return ShapSummary(
                available=False,
                warning=(
                    f"SHAP is not enabled for {estimator_name}. "
                    "Permutation importance remains available."
                ),
            )

        local_position = min(
            local_row_position,
            len(x_holdout) - 1,
        )

        explain_positions = self._sample_positions(
            row_count=len(x_holdout),
            sample_size=min(
                sample_size,
                self.settings.explainability_max_sample_size,
            ),
            required_position=local_position,
        )

        background_positions = self._sample_positions(
            row_count=len(x_holdout),
            sample_size=min(
                background_size,
                self.settings.explainability_max_background_size,
            ),
            required_position=None,
        )

        x_explain_raw = x_holdout.iloc[explain_positions]
        x_background_raw = x_holdout.iloc[background_positions]

        try:
            x_explain = self._dense_matrix(preprocessor.transform(x_explain_raw))
            x_background = self._dense_matrix(preprocessor.transform(x_background_raw))
        except Exception as exc:
            return ShapSummary(
                available=False,
                explainer_type=explainer_type,
                warning=(f"Failed to transform SHAP data: {exc}"),
            )

        feature_names = self._feature_names(
            preprocessor=preprocessor,
            feature_count=x_explain.shape[1],
        )

        source_features = [
            self._source_feature_name(
                transformed_feature=feature_name,
                raw_features=[str(column) for column in x_holdout.columns],
            )
            for feature_name in feature_names
        ]

        try:
            if explainer_type == "TreeExplainer":
                explainer = shap.TreeExplainer(
                    estimator,
                    data=x_background,
                    feature_names=feature_names,
                )
                explanation = explainer(
                    x_explain,
                    check_additivity=False,
                )
            else:
                explainer = shap.LinearExplainer(
                    estimator,
                    x_background,
                    feature_names=feature_names,
                )
                explanation = explainer(x_explain)
        except Exception as exc:
            return ShapSummary(
                available=False,
                explainer_type=explainer_type,
                warning=(f"{explainer_type} failed: {exc}"),
            )

        (
            shap_values,
            base_values,
            output_name,
        ) = self._select_output(
            explanation=explanation,
            estimator=estimator,
            positive_class=positive_class,
        )

        if shap_values is None:
            return ShapSummary(
                available=False,
                explainer_type=explainer_type,
                warning=("Unable to select a SHAP output."),
            )

        transformed_importance = self._transformed_importance(
            shap_values=shap_values,
            feature_names=feature_names,
            source_features=source_features,
        )

        source_importance = self._source_importance(
            shap_values=shap_values,
            source_features=source_features,
        )

        beeswarm_points = self._beeswarm_points(
            shap_values=shap_values,
            transformed_values=x_explain,
            feature_names=feature_names,
            source_features=source_features,
            transformed_importance=(transformed_importance),
        )

        local_sample_position = explain_positions.index(local_position)

        local_explanation = self._local_explanation(
            pipeline=pipeline,
            x_holdout=x_holdout,
            x_explain_raw=x_explain_raw,
            transformed_values=x_explain,
            shap_values=shap_values,
            base_values=base_values,
            feature_names=feature_names,
            source_features=source_features,
            output_name=output_name,
            local_holdout_position=local_position,
            local_sample_position=local_sample_position,
        )

        return ShapSummary(
            available=True,
            explainer_type=explainer_type,
            output_name=output_name,
            transformed_feature_importance=(transformed_importance),
            source_feature_importance=(source_importance),
            beeswarm_points=beeswarm_points,
            local_explanation=local_explanation,
        )

    def _select_output(
        self,
        explanation,
        estimator,
        positive_class: str | None,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray,
        str,
    ]:
        values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values)

        classes = [
            str(value)
            for value in getattr(
                estimator,
                "classes_",
                [],
            )
        ]

        output_name = positive_class or (classes[-1] if classes else "model_output")

        if values.ndim == 3:
            output_index = self._output_index(
                classes=classes,
                requested_output=output_name,
                output_count=values.shape[2],
            )

            values = values[:, :, output_index]

            if base_values.ndim == 2:
                base_values = base_values[
                    :,
                    output_index,
                ]
            elif base_values.ndim == 1 and len(base_values) == values.shape[0]:
                pass
            elif base_values.ndim == 1:
                base_values = np.repeat(
                    base_values[output_index],
                    values.shape[0],
                )

            if classes:
                output_name = classes[output_index]

        elif values.ndim == 2:
            if base_values.ndim == 0:
                base_values = np.repeat(
                    float(base_values),
                    values.shape[0],
                )
            elif base_values.ndim == 1 and len(base_values) != values.shape[0]:
                base_values = np.repeat(
                    float(base_values[-1]),
                    values.shape[0],
                )
        else:
            return None, np.asarray([]), output_name

        return (
            np.asarray(values, dtype=float),
            np.asarray(base_values, dtype=float),
            output_name,
        )

    def _output_index(
        self,
        classes: list[str],
        requested_output: str,
        output_count: int,
    ) -> int:
        if requested_output in classes:
            return classes.index(requested_output)

        return max(output_count - 1, 0)

    def _transformed_importance(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
        source_features: list[str],
    ) -> list[ExplainabilityImportanceItem]:
        mean_absolute_values = np.mean(
            np.abs(shap_values),
            axis=0,
        )

        items = [
            ExplainabilityImportanceItem(
                feature=feature_names[index],
                source_feature=source_features[index],
                importance_mean=round(
                    float(value),
                    6,
                ),
            )
            for index, value in enumerate(mean_absolute_values)
        ]

        items.sort(
            key=lambda item: item.importance_mean,
            reverse=True,
        )

        return items

    def _source_importance(
        self,
        shap_values: np.ndarray,
        source_features: list[str],
    ) -> list[ExplainabilityImportanceItem]:
        source_contributions: dict[
            str,
            np.ndarray,
        ] = defaultdict(
            lambda: np.zeros(
                shap_values.shape[0],
                dtype=float,
            )
        )

        for index, source_feature in enumerate(source_features):
            source_contributions[source_feature] += shap_values[:, index]

        items = []

        for source_feature, contributions in source_contributions.items():
            items.append(
                ExplainabilityImportanceItem(
                    feature=source_feature,
                    source_feature=source_feature,
                    importance_mean=round(
                        float(np.mean(np.abs(contributions))),
                        6,
                    ),
                )
            )

        items.sort(
            key=lambda item: item.importance_mean,
            reverse=True,
        )

        return items

    def _beeswarm_points(
        self,
        shap_values: np.ndarray,
        transformed_values: np.ndarray,
        feature_names: list[str],
        source_features: list[str],
        transformed_importance: list[ExplainabilityImportanceItem],
    ) -> list[ShapBeeswarmPoint]:
        maximum_features = self.settings.explainability_max_beeswarm_features

        top_feature_names = {item.feature for item in transformed_importance[:maximum_features]}

        points = []

        for feature_index, feature_name in enumerate(feature_names):
            if feature_name not in top_feature_names:
                continue

            for row_position in range(shap_values.shape[0]):
                points.append(
                    ShapBeeswarmPoint(
                        feature=feature_name,
                        source_feature=(source_features[feature_index]),
                        row_position=row_position,
                        shap_value=round(
                            float(
                                shap_values[
                                    row_position,
                                    feature_index,
                                ]
                            ),
                            6,
                        ),
                        feature_value=self._safe_float(
                            transformed_values[
                                row_position,
                                feature_index,
                            ]
                        ),
                    )
                )

        return points

    def _local_explanation(
        self,
        pipeline: Pipeline,
        x_holdout: pd.DataFrame,
        x_explain_raw: pd.DataFrame,
        transformed_values: np.ndarray,
        shap_values: np.ndarray,
        base_values: np.ndarray,
        feature_names: list[str],
        source_features: list[str],
        output_name: str,
        local_holdout_position: int,
        local_sample_position: int,
    ) -> ShapLocalExplanation:
        raw_row = x_holdout.iloc[[local_holdout_position]]

        prediction = pipeline.predict(raw_row)[0]

        probabilities = self._probabilities(
            pipeline=pipeline,
            raw_row=raw_row,
        )

        local_values = shap_values[local_sample_position]

        contribution_indices = np.argsort(np.abs(local_values))[::-1][
            : self.settings.explainability_max_local_features
        ]

        contributions = [
            ShapLocalContribution(
                feature=feature_names[index],
                source_feature=source_features[index],
                feature_value=self._safe_float(
                    transformed_values[
                        local_sample_position,
                        index,
                    ]
                ),
                shap_value=round(
                    float(local_values[index]),
                    6,
                ),
                direction=("positive" if local_values[index] >= 0 else "negative"),
            )
            for index in contribution_indices
        ]

        base_value = self._base_value(
            base_values=base_values,
            row_position=local_sample_position,
        )

        explained_output_value = None

        if base_value is not None:
            explained_output_value = round(
                base_value + float(np.sum(local_values)),
                6,
            )

        dataset_index = str(x_explain_raw.index[local_sample_position])

        raw_record = {
            str(key): self._json_value(value) for key, value in raw_row.iloc[0].to_dict().items()
        }

        return ShapLocalExplanation(
            holdout_row_position=(local_holdout_position),
            dataset_index=dataset_index,
            output_name=output_name,
            base_value=base_value,
            explained_output_value=(explained_output_value),
            prediction=self._json_value(prediction),
            probabilities=probabilities,
            raw_record=raw_record,
            contributions=contributions,
        )

    def _probabilities(
        self,
        pipeline: Pipeline,
        raw_row: pd.DataFrame,
    ) -> dict[str, float] | None:
        if not hasattr(
            pipeline,
            "predict_proba",
        ):
            return None

        estimator = pipeline.named_steps["model"]

        if not hasattr(estimator, "classes_"):
            return None

        try:
            values = pipeline.predict_proba(raw_row)[0]
        except Exception:
            return None

        return {
            str(label): round(
                float(values[index]),
                6,
            )
            for index, label in enumerate(estimator.classes_)
        }

    def _feature_names(
        self,
        preprocessor,
        feature_count: int,
    ) -> list[str]:
        try:
            names = [str(value) for value in (preprocessor.get_feature_names_out())]
        except Exception:
            names = [f"feature_{index}" for index in range(feature_count)]

        if len(names) != feature_count:
            return [f"feature_{index}" for index in range(feature_count)]

        return names

    def _source_feature_name(
        self,
        transformed_feature: str,
        raw_features: list[str],
    ) -> str:
        if "__" in transformed_feature:
            transformer_name, remainder = transformed_feature.split(
                "__",
                1,
            )
        else:
            transformer_name = ""
            remainder = transformed_feature

        if transformer_name.startswith("text_"):
            transformer_slug = transformer_name[len("text_") :]

            for raw_feature in raw_features:
                if self._slugify(raw_feature) == transformer_slug:
                    return raw_feature

        sorted_features = sorted(
            raw_features,
            key=len,
            reverse=True,
        )

        for raw_feature in sorted_features:
            if (
                remainder == raw_feature
                or remainder.startswith(f"{raw_feature}_")
                or remainder.startswith(f"{raw_feature}__")
            ):
                return raw_feature

        return remainder

    def _sample_positions(
        self,
        row_count: int,
        sample_size: int,
        required_position: int | None,
    ) -> list[int]:
        actual_size = min(
            row_count,
            sample_size,
        )

        positions = np.linspace(
            0,
            row_count - 1,
            actual_size,
            dtype=int,
        ).tolist()

        positions = list(dict.fromkeys(positions))

        if required_position is not None and required_position not in positions:
            if positions:
                positions[-1] = required_position
            else:
                positions.append(required_position)

            positions = sorted(set(positions))

        return positions

    def _dense_matrix(
        self,
        matrix,
    ) -> np.ndarray:
        if sparse.issparse(matrix):
            estimated_cells = matrix.shape[0] * matrix.shape[1]

            if estimated_cells > 4_000_000:
                raise ValueError("SHAP transformed matrix is too large.")

            matrix = matrix.toarray()

        dense_matrix = np.asarray(
            matrix,
            dtype=float,
        )

        if dense_matrix.ndim != 2:
            raise ValueError("SHAP matrix must be two-dimensional.")

        return dense_matrix

    def _base_value(
        self,
        base_values: np.ndarray,
        row_position: int,
    ) -> float | None:
        if base_values.ndim == 0:
            return self._safe_float(base_values)

        if len(base_values) == 0:
            return None

        position = min(
            row_position,
            len(base_values) - 1,
        )

        return self._safe_float(base_values[position])

    def _json_value(
        self,
        value: Any,
    ) -> Any:
        if pd.isna(value):
            return None

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        return value

    def _safe_float(
        self,
        value: object,
    ) -> float | None:
        try:
            float_value = float(cast(Any, value))
        except (TypeError, ValueError):
            return None

        if not math.isfinite(float_value):
            return None

        return round(float_value, 6)

    def _slugify(
        self,
        value: str,
    ) -> str:
        return "".join(
            character if character.isalnum() else "_" for character in value.lower()
        ).strip("_")
