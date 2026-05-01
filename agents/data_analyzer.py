class DataAnalyzer:
    def detect_target(self, df):
        # simple heuristic: last column
        return df.columns[-1]

    def analyze(self, df, target_col=None):
        if target_col is None:
            target_col = self.detect_target(df)

        summary = {}
        summary["target_col"] = target_col

        summary["num_rows"] = df.shape[0]
        summary["num_cols"] = df.shape[1]

        summary["missing"] = df.isnull().sum().to_dict()

        summary["numeric_cols"] = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        summary["categorical_cols"] = df.select_dtypes(include=['object']).columns.tolist()

        # Problem type
        if df[target_col].nunique() < 10:
            summary["problem_type"] = "classification"
        else:
            summary["problem_type"] = "regression"

        return summary