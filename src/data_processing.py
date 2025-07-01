import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
class CustomerAggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Creates aggregated numerical features per CustomerId
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        # Ensure a copy is used to avoid modifying original data
        df = X.copy()

        # Group by CustomerId and aggregate
        agg_df = df.groupby('CustomerId').agg(
            TotalTransactionAmount=('Amount', 'sum'),
            AvgTransactionAmount=('Amount', 'mean'),
            TransactionCount=('Amount', 'count'),
            StdTransactionAmount=('Amount', 'std')
        ).reset_index()

        return agg_df

class TimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts hour, day, month, and year from TransactionStartTime column.
    Assumes TransactionStartTime is already a datetime column.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Ensure datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)

        # Extract features
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year

        return df[['CustomerId', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']]

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encodes selected categorical columns and returns dummy variables.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # One-hot encode specified columns
        encoded = pd.get_dummies(df[self.columns], prefix=self.columns)

        # Keep CustomerId for joining later
        encoded['CustomerId'] = df['CustomerId'].values

        return encoded

def build_numeric_pipeline(numeric_columns):
    """
    Builds a pipeline that imputes and scales numeric features.
    """
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),       # Fill missing values with mean
        ('scaler', StandardScaler())                       # Scale features to N(0, 1)
    ])

    # ColumnTransformer applies it only to numeric columns
    full_pipeline = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_columns)
    ])

    return full_pipeline
