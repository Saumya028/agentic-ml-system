import numpy as np

class InsightAgent:
    def generate_report(self, df, summary, decision):
        report = []

        target = summary["target_col"]
        report.append(f"Problem Type: {summary['problem_type']} (target = {target})\n")

        # -------------------
        # Missing Values Analysis
        # -------------------
        report.append("Missing Value Analysis:")

        missing = summary["missing"]
        total_rows = summary["num_rows"]

        for col, val in missing.items():
            if val > 0:
                percent = (val / total_rows) * 100

                if percent > 80:
                    report.append(f"- DROP column '{col}' ({percent:.1f}% missing)")
                elif percent > 0:
                    report.append(f"- IMPUTE '{col}' with median/mode ({percent:.1f}% missing)")

        # -------------------
        # Categorical Handling
        # -------------------
        report.append("\nCategorical Features:")

        for col in summary["categorical_cols"]:
            unique_vals = df[col].nunique()

            if unique_vals < 10:
                report.append(f"- One-hot encode '{col}'")
            else:
                report.append(f"- Label encode '{col}' (high cardinality)")

        # -------------------
        # Skewness Detection
        # -------------------
        report.append("\nSkewness Analysis:")

        numeric_cols = summary["numeric_cols"]

        for col in numeric_cols:
            if col == target:
                continue

            skew = df[col].skew()

            if skew > 1:
                report.append(f"- Apply log transform to '{col}' (skew={skew:.2f})")

        # -------------------
        # Scaling
        # -------------------
        if decision["scaling"]:
            report.append("\nScaling:")
            report.append("- Apply StandardScaler")

        # -------------------
        # Model Recommendation
        # -------------------
        report.append("\nModel Recommendation:")
        report.append(f"- Models to try: {decision['models']}")

        # Distribution insight
        report.append("\nDistribution Impact:")

        for col in numeric_cols:
            if col == target:
                continue

            skew = df[col].skew()

            if skew > 1:
                report.append(f"- '{col}' is highly skewed → may hurt linear models")

        return "\n".join(report)