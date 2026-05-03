import requests
import os
from dotenv import load_dotenv

load_dotenv()

class CodeGenerationAgent:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found. Check your .env file.")

    def generate_code(self, summary, decision, best_model):
        # Extract structured info
        target = summary["target_col"]
        problem_type = summary["problem_type"]
        numeric_cols = summary["numeric_cols"]
        categorical_cols = summary["categorical_cols"]
        missing = summary["missing"]

        # Identify high-missing columns
        high_missing = [col for col, val in missing.items() if val > 0.8 * summary["num_rows"]]

        # Identify skewed columns (passed from insight logic ideally)
        skewed_cols = []  # You can later pass this dynamically

        prompt = f"""
You are a senior Machine Learning Engineer.

Generate a COMPLETE, production-ready Python script (ONLY CODE, NO EXPLANATION).

The code MUST strictly follow the dataset analysis provided.

-------------------------
DATASET ANALYSIS:
-------------------------
Target Column: {target}
Problem Type: {problem_type}

Numeric Columns: {numeric_cols}
Categorical Columns: {categorical_cols}

Columns with HIGH missing values (DROP): {high_missing}

-------------------------
PREPROCESSING RULES:
-------------------------
1. Drop columns with high missing values
2. Use ColumnTransformer
3. Numeric pipeline:
   - Median imputation
   - StandardScaler
4. Categorical pipeline:
   - Most frequent imputation
   - OneHotEncoder (handle_unknown='ignore')

5. If skewed features exist → apply log1p transform

-------------------------
VISUALIZATION:
-------------------------
- Histogram for target
- Correlation heatmap

-------------------------
MODEL:
-------------------------
Use: {best_model}

-------------------------
REQUIREMENTS:
-------------------------
- Use sklearn Pipeline
- Use train_test_split
- Evaluate properly:
    - regression → R2 score
    - classification → accuracy + F1
- Clean modular code
- Must run without errors
- No placeholders
- Use exact column names provided

IMPORTANT:
- DO NOT generate generic code
- DO NOT ignore preprocessing details
- STRICTLY follow given dataset structure
- Use ColumnTransformer properly

OUTPUT:
ONLY PYTHON CODE
"""

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                }
            )

            response_json = response.json()

            print("\n===== API RESPONSE =====")
            print(response_json)

            # Error handling
            if "error" in response_json:
                return f"API Error: {response_json['error']}"

            if "choices" not in response_json:
                return f"Unexpected response: {response_json}"

            content = response_json["choices"][0]["message"]["content"]

            # Clean markdown formatting if present
            if "```" in content:
                content = content.split("```")[-2]

            return content

        except Exception as e:
            return f"Exception occurred: {str(e)}"