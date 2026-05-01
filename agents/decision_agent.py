class DecisionAgent:
    def decide(self, summary):
        decision = {}

        total_missing = sum(summary["missing"].values())

        # Imputation
        decision["imputation"] = "median" if total_missing > 0 else None

        # Encoding
        if len(summary["categorical_cols"]) > 0:
            decision["encoding"] = "onehot"
        else:
            decision["encoding"] = None

        # Scaling
        decision["scaling"] = True

        # Models
        if summary["problem_type"] == "classification":
            decision["models"] = ["logistic", "random_forest"]
        else:
            decision["models"] = ["linear", "random_forest"]

        return decision