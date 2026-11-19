# Student Performance Prediction & Analysis System

A Python + Machine Learning project that estimates student exam marks from academic and study-habit indicators and displays an interactive analytics dashboard.

## Features
- Predicts estimated final marks using a Random Forest Regressor.
- Displays a simple performance band: Good, Average, or Needs Academic Support.
- Visualizes study hours vs marks, marks distribution, and attendance groups.
- Includes a reproducible sample dataset and model-training script.
- Streamlit-based interactive dashboard.

## Technology Stack
Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly, Joblib.

## Project Structure
```text
Student_Performance_Prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── student_performance.csv
└── models/
    └── (model is created automatically on first run)
```

## Run on Windows (PowerShell)

1. Extract the ZIP file and open the `Student_Performance_Prediction` folder in VS Code.
2. Open the VS Code terminal and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The dashboard should open in your browser. If it does not, open the local URL printed in the terminal (usually `http://localhost:8501`).

## Train the Model Separately (Optional)

```powershell
python train_model.py
```

The trained model will be saved automatically in the `models` folder.

## Dataset
The included `student_performance.csv` contains 600 **synthetic sample records**. It is intended for learning and demonstration, not real student assessment. Replace it with an appropriately licensed, consented dataset before making real-world claims.

## Model Evaluation
When `train_model.py` runs, it prints:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² score

These metrics describe performance on the held-out test split of the included synthetic dataset. Do not present them as clinical, institutional, or real-world validation.

## Disclaimer
This project is an educational prototype. Predicted marks are estimates and should not be used as official grades or for consequential decisions about students.
