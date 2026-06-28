from datetime import datetime

from pydantic import BaseModel, Field


class MissingColumnSummary(BaseModel):
    name: str = Field(description="Column name")
    inferred_type: str = Field(description="Inferred column type")
    missing_count: int = Field(description="Missing value count")
    missing_ratio: float = Field(description="Missing value ratio")


class MissingAnalysisResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    total_missing_cells: int = Field(description="Total missing cell count")
    missing_cell_ratio: float = Field(description="Missing cell ratio")
    columns: list[MissingColumnSummary] = Field(description="Missing analysis by column")


class DuplicateAnalysisResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    duplicate_row_count: int = Field(description="Duplicate row count")
    duplicate_row_ratio: float = Field(description="Duplicate row ratio")


class NumericColumnStatistics(BaseModel):
    name: str = Field(description="Column name")
    count: int = Field(description="Non-null numeric value count")
    mean: float | None = Field(description="Mean value")
    std: float | None = Field(description="Standard deviation")
    min: float | None = Field(description="Minimum value")
    q25: float | None = Field(description="25th percentile")
    median: float | None = Field(description="Median value")
    q75: float | None = Field(description="75th percentile")
    max: float | None = Field(description="Maximum value")
    skewness: float | None = Field(description="Skewness")
    kurtosis: float | None = Field(description="Kurtosis")


class NumericStatisticsResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    columns: list[NumericColumnStatistics] = Field(description="Numeric statistics")


class OutlierColumnSummary(BaseModel):
    name: str = Field(description="Column name")
    method: str = Field(description="Outlier detection method")
    lower_bound: float | None = Field(description="Lower outlier bound")
    upper_bound: float | None = Field(description="Upper outlier bound")
    outlier_count: int = Field(description="Outlier value count")
    outlier_ratio: float = Field(description="Outlier value ratio")


class OutlierAnalysisResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    columns: list[OutlierColumnSummary] = Field(description="Outlier analysis by column")


class CorrelationPair(BaseModel):
    column_x: str = Field(description="First column name")
    column_y: str = Field(description="Second column name")
    correlation: float = Field(description="Pearson correlation value")


class CorrelationAnalysisResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    method: str = Field(description="Correlation method")
    columns: list[str] = Field(description="Numeric columns used in correlation")
    matrix: dict[str, dict[str, float | None]] = Field(description="Correlation matrix")
    strongest_pairs: list[CorrelationPair] = Field(description="Strongest correlation pairs")


class EdaSummaryResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    row_count: int = Field(description="Dataset row count")
    column_count: int = Field(description="Dataset column count")
    data_quality_score: float = Field(description="Data quality score from 0 to 100")
    data_quality_grade: str = Field(description="Data quality grade")
    missing: MissingAnalysisResponse = Field(description="Missing value analysis")
    duplicates: DuplicateAnalysisResponse = Field(description="Duplicate row analysis")
    numeric_statistics: NumericStatisticsResponse = Field(description="Numeric statistics")
    outliers: OutlierAnalysisResponse = Field(description="Outlier analysis")
    correlation: CorrelationAnalysisResponse = Field(description="Correlation analysis")
    recommendations: list[str] = Field(description="Rule-based EDA recommendations")
    generated_at: datetime = Field(description="EDA generation timestamp")
