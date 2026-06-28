from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateTimeFeatureExtractor(
    BaseEstimator,
    TransformerMixin,
):
    def fit(
        self,
        x_data: Any,
        y_data: Any = None,
    ) -> "DateTimeFeatureExtractor":
        # 保存輸入欄位名稱，讓 pipeline 能產生可讀的 feature names。
        if isinstance(x_data, pd.DataFrame):
            self.feature_names_in_ = np.asarray(
                [str(column) for column in x_data.columns],
                dtype=object,
            )
        else:
            array = np.asarray(x_data)
            column_count = 1 if array.ndim == 1 else array.shape[1]

            self.feature_names_in_ = np.asarray(
                [f"datetime_{index}" for index in range(column_count)],
                dtype=object,
            )

        return self

    def transform(
        self,
        x_data: Any,
    ) -> np.ndarray:
        frame = self._to_dataframe(x_data)
        transformed_columns = []

        for column in frame.columns:
            values = pd.to_datetime(
                frame[column],
                errors="coerce",
            )

            transformed_columns.extend(
                [
                    values.dt.year.to_numpy(dtype=float),
                    values.dt.month.to_numpy(dtype=float),
                    values.dt.day.to_numpy(dtype=float),
                    values.dt.dayofweek.to_numpy(dtype=float),
                    values.dt.quarter.to_numpy(dtype=float),
                    values.dt.is_month_end.to_numpy(dtype=float),
                ]
            )

        if not transformed_columns:
            return np.empty(
                (len(frame), 0),
                dtype=float,
            )

        return np.column_stack(transformed_columns)

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        features = (
            np.asarray(input_features, dtype=object)
            if input_features is not None
            else self.feature_names_in_
        )

        suffixes = [
            "year",
            "month",
            "day",
            "dayofweek",
            "quarter",
            "is_month_end",
        ]

        return np.asarray(
            [f"{feature}__{suffix}" for feature in features for suffix in suffixes],
            dtype=object,
        )

    def _to_dataframe(
        self,
        x_data: Any,
    ) -> pd.DataFrame:
        if isinstance(x_data, pd.DataFrame):
            return x_data.copy()

        array = np.asarray(x_data)

        if array.ndim == 1:
            array = array.reshape(-1, 1)

        return pd.DataFrame(
            array,
            columns=self.feature_names_in_,
        )


class TextSeriesCleaner(
    BaseEstimator,
    TransformerMixin,
):
    def fit(
        self,
        x_data: Any,
        y_data: Any = None,
    ) -> "TextSeriesCleaner":
        return self

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        # Pipeline 需要這個方法才能向後傳遞 feature names。
        if input_features is None:
            return np.asarray(
                ["text"],
                dtype=object,
            )

        return np.asarray(
            input_features,
            dtype=object,
        )

    def transform(
        self,
        x_data: Any,
    ) -> list[str]:
        # TfidfVectorizer 需要一維字串序列，這裡統一補空值與轉型。
        if isinstance(x_data, pd.Series):
            series = x_data
        elif isinstance(x_data, pd.DataFrame):
            series = x_data.iloc[:, 0]
        else:
            array = np.asarray(x_data)

            if array.ndim > 1:
                array = array[:, 0]

            series = pd.Series(array)

        return series.fillna("").astype(str).tolist()
