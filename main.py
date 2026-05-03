import pandas as pd

from agents.data_analyzer import DataAnalyzer
from agents.decision_agent import DecisionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.model_agent import ModelAgent
from agents.evaluation_agent import EvaluationAgent
from agents.insight_agent import InsightAgent
from agents.code_generation_agent import CodeGenerationAgent


from utils.helpers import save_memory
from config import TARGET_COLUMN, ITERATIONS

# Load dataset
df = pd.read_csv("train.csv")

# Initialize agents
analyzer = DataAnalyzer()
decision_agent = DecisionAgent()
preprocessor = PreprocessingAgent()
model_agent = ModelAgent()
evaluator = EvaluationAgent()
insight_agent = InsightAgent()
code_agent = CodeGenerationAgent()

for i in range(ITERATIONS):
    print(f"\n--- Iteration {i+1} ---")

    summary = analyzer.analyze(df, TARGET_COLUMN if TARGET_COLUMN else None)
    target_col = summary["target_col"]
    print("Summary:", summary)

    decision = decision_agent.decide(summary)
    print("Decision:", decision)

    X, y, pipeline = preprocessor.process(df, target_col, decision)

    models = model_agent.train(X, y, summary["problem_type"], decision["models"])

    best_model, results = evaluator.evaluate(X, y, models, summary["problem_type"])

    print("Results:", results)
    print("Best Model:", best_model)

    save_memory({
        "summary": summary,
        "decision": decision,
        "results": results,
        "best_model": best_model
    })

    report = insight_agent.generate_report(df, summary, decision)
    print("\n===== DATA REPORT =====")
    print(report)

generated_code = code_agent.generate_code(summary, decision, best_model)

print("\n===== GENERATED ML CODE =====\n")
print(generated_code)
