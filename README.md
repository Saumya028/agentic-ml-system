# 🤖 Agentic ML System

An intelligent **Agent-based AutoML system** that analyzes raw CSV datasets, performs automated data understanding, recommends preprocessing strategies, selects optimal models, and generates **production-ready Python code** using LLMs.

---

## 🚀 Features

### 🧠 1. Data Analysis Agent
- Detects:
  - Problem type (Classification / Regression)
  - Missing values
  - Feature types (Numerical / Categorical)
  - Dataset structure

---

### ⚙️ 2. Decision Agent
- Recommends:
  - Imputation strategy
  - Encoding method
  - Feature scaling
  - Model selection

---

### 📊 3. Insight Agent (Explainability Layer)
- Generates a **human-readable report**
- Identifies:
  - Missing data issues
  - Skewed features
  - High-risk columns
  - Performance bottlenecks
- Suggests:
  - Data cleaning strategies
  - Feature engineering steps

---

### 🧪 4. Model Training & Evaluation
- Trains multiple models:
  - Linear / Logistic Regression
  - Random Forest
  - XGBoost (optional)
- Evaluates using:
  - R² (Regression)
  - Accuracy / F1 (Classification)
- Selects **best-performing model**

---

### 🤖 5. Code Generation Agent (LLM-Powered)
- Uses OpenRouter API
- Generates **complete Python pipeline code**:
  - Data preprocessing (ColumnTransformer)
  - Feature engineering
  - Visualization (histograms, heatmaps)
  - Model training
  - Evaluation
- Fully aligned with dataset analysis

---

### 🧠 6. Memory Module
- Stores previous runs
- Enables future learning & optimization

---

## 🏗️ Project Architecture
CSV Input
↓
Data Analyzer Agent
↓
Decision Agent
↓
Insight Agent (Report)
↓
Preprocessing + Model Training
↓
Evaluation Agent
↓
Code Generation Agent (LLM)
↓
Final Output:
✔ Data Report
✔ Best Model
✔ Generated Python Code

---

## 🛠️ Tech Stack

- Python
- pandas, numpy
- scikit-learn
- xgboost
- OpenRouter (LLM API)
- dotenv

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/agentic-ml-system.git
cd agentic-ml-system

pip install -r requirements.txt

🔐 Environment Setup

Create a .env file in root:

OPENROUTER_API_KEY=your_api_key_here
▶️ Run the Project
python main.py
📊 Output

The system generates:

✔ Dataset analysis report
✔ Recommended preprocessing steps
✔ Best ML model
✔ Model performance metrics
✔ Generated Python code for full pipeline
⚠️ Limitations
Heuristic-based decision system (not full AutoML yet)
LLM-generated code may need minor validation
No hyperparameter tuning (yet)
🚀 Future Improvements
🔥 Hyperparameter tuning agent
🧠 Meta-learning (dataset similarity)
📊 SHAP explainability
📓 Jupyter notebook generation
🔁 Self-correcting LLM loop
💡 Author

Saumya
AI & ML Engineering Student

⭐ If you like this project

Give it a star ⭐ and feel free to contribute!


---