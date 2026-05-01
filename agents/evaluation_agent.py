from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE

class EvaluationAgent:
    def evaluate(self, X, y, models, problem_type):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            if problem_type == "classification":
                score = accuracy_score(y_test, preds)
            else:
                score = r2_score(y_test, preds)

            results[name] = score

        best_model = max(results, key=results.get)

        return best_model, results