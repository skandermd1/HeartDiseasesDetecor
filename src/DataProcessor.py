import pandas as pd
from sklearn.impute import SimpleImputer


class Processor:
    def HandleMissingValues(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=['number']).columns

        if numeric_cols.empty:
            return df

        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        return df