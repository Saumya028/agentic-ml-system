from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

class ModelAgent:
    def train(self, X, y, problem_type, models):
        trained_models = {}

        for model_name in models:
            if problem_type == "classification":
                if model_name == "logistic":
                    model = LogisticRegression(max_iter=1000)
                elif model_name == "random_forest":
                    model = RandomForestClassifier()
                elif model_name == "xgboost":
                    model = XGBClassifier()
            else:
                if model_name == "linear":
                    model = LinearRegression()
                elif model_name == "random_forest":
                    model = RandomForestRegressor()
                elif model_name == "xgboost":
                    model = XGBRegressor()

            model.fit(X, y)
            trained_models[model_name] = model

        return trained_models