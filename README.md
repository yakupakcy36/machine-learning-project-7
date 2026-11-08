# Credit Card Fraud Detection

Small learning project. The notebook `credit_card_fraud_detection.ipynb` trains logistic regression on undersampled data.

Data: Credit card transactions (creditcard.csv), highly imbalanced fraud labels. The data files are not included in this repo, so you need to download them yourself.

## Benchmark script

`src/benchmark.py` compares HistGradientBoosting with class weights, balanced random forest and undersampled logistic regression with 5-fold cross-validation.
It does not use the real dataset. It runs on synthetic data generated with numpy/scikit-learn (6000 samples, 30 features, about 3.2% fraud), so the numbers below only compare the models on that data and say nothing about the real dataset.

## How to run

```
pip install numpy scikit-learn jupyter
jupyter notebook
python src/benchmark.py
```

## Results (5-fold CV, mean)

| Model | roc_auc | pr_auc | f1 | precision | recall |
|---|---|---|---|---|---|
| Cost-Sensitive HistGradientBoosting | 0.9493 | 0.7666 | 0.7112 | 0.8864 | 0.5939 |
| Balanced Subsample Random Forest | 0.934 | 0.6862 | 0.6018 | 0.75 | 0.5025 |
| Undersampled Logistic Regression | 0.8938 | 0.5042 | 0.2522 | 0.1493 | 0.8122 |

Full numbers are in `benchmarks/benchmark_results.json`.

## Files

- `credit_card_fraud_detection.ipynb`: notebook
- `src/benchmark.py`: benchmark script
- `benchmarks/benchmark_results.json`: results
- `assets/benchmark_chart.svg`: chart of the results
