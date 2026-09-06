<<<<<<< HEAD
# NLP Sentiment Analysis Benchmark

A reproducible NLP benchmarking project comparing three approaches to binary sentiment analysis on the **NLTK Movie Reviews dataset**:

* **TF-IDF + Logistic Regression** — traditional supervised machine learning
* **PyTorch LSTM** — deep learning with learned word embeddings
* **VADER** — lexicon and rule-based sentiment analysis

The project evaluates predictive performance together with training cost, inference latency, error behaviour, and practical deployment trade-offs.

---

## Project Overview

Sentiment analysis is a Natural Language Processing task used to determine whether a piece of text expresses a positive or negative sentiment.

This project benchmarks three different NLP approaches on the same dataset and test partition to understand the trade-offs between:

* Predictive performance
* Model complexity
* Training cost
* Inference latency
* Error behaviour
* Practical deployment suitability

The results demonstrate that a more complex deep-learning model does not necessarily provide better performance, particularly when the available labelled dataset is relatively small.

---

## Models Compared

### 1. TF-IDF + Logistic Regression

A traditional supervised machine-learning pipeline.

```text
Raw Review
    ↓
Text Preprocessing
    ↓
TF-IDF Features
    ↓
Logistic Regression
    ↓
Sentiment Prediction
```

TF-IDF converts each cleaned review into weighted lexical features, including unigram and bigram information. Logistic Regression then performs binary sentiment classification.

---

### 2. PyTorch LSTM

A sequence-based deep-learning model using learned word embeddings.

```text
Raw Review
    ↓
Text Preprocessing
    ↓
Vocabulary Construction
    ↓
Integer Encoding
    ↓
Padding / Truncation
    ↓
Embedding Layer
    ↓
LSTM
    ↓
Dropout
    ↓
Linear Classification Layer
    ↓
Sentiment Prediction
```

The vocabulary is constructed from training data, reviews are converted to integer sequences, and sequences are padded or truncated to a controlled length.

The training pipeline includes validation monitoring and early stopping.

---

### 3. VADER

A training-free sentiment-analysis approach based on a sentiment lexicon and rule-based scoring.

```text
Raw Review
    ↓
VADER Sentiment Analyzer
    ↓
Compound Sentiment Score
    ↓
Positive / Negative Prediction
```

VADER does not require dataset-specific model training.

---

# Dataset

The project uses the **NLTK Movie Reviews dataset**.

### Dataset characteristics

| Property         | Value |
| ---------------- | ----: |
| Total reviews    | 2,000 |
| Positive reviews | 1,000 |
| Negative reviews | 1,000 |
| Training reviews | 1,600 |
| Test reviews     |   400 |

A stratified train/test split is used so that both sentiment classes remain represented consistently.

---

# Preprocessing

The preprocessing pipeline aims to preserve sentiment-bearing information while reducing unnecessary textual variation.

### Processing steps

1. Convert text to lowercase
2. Normalize contracted negation
3. Remove URLs and HTML
4. Remove punctuation and numbers
5. Tokenize text
6. Remove stop words
7. Preserve sentiment-bearing negations:

   * `no`
   * `nor`
   * `not`
8. Remove one-character tokens
9. Normalize whitespace

Exact duplicate reviews are handled carefully so that reviews with conflicting sentiment labels are not silently discarded.

---

# Benchmark Results

The following results come from the latest successful local benchmark run.

| Model                            |   Accuracy |   Macro F1 | Positive-Class F1 | Preparation + Training | Test Inference | Inference / Review |
| -------------------------------- | ---------: | ---------: | ----------------: | ---------------------: | -------------: | -----------------: |
| **TF-IDF + Logistic Regression** | **84.75%** | **84.74%** |        **85.09%** |           **2.8864 s** |   **0.1661 s** |       **0.415 ms** |
| PyTorch LSTM                     |     52.75% |     52.62% |            50.13% |             531.1268 s |       0.7209 s |           1.802 ms |
| VADER                            |     64.75% |     63.79% |            69.68% |               0.0000 s |       1.8867 s |           4.717 ms |

> Timing measurements are environment-dependent. Accuracy and F1 values are the primary predictive-performance metrics.

---

# Key Findings

## Logistic Regression is the strongest model

TF-IDF + Logistic Regression achieved the best overall performance:

* **84.75% Accuracy**
* **84.74% Macro F1**
* **85.09% Positive-Class F1**

It outperformed both the LSTM and VADER approaches on the current benchmark.

---

## Logistic Regression vs LSTM

The Accuracy gap is:

**84.75% − 52.75% = 32.00 percentage points**

The Macro F1 gap is:

**84.74% − 52.62% = 22.12 percentage points**

The current LSTM therefore does not justify its substantially higher computational cost under the present configuration.

---

## Logistic Regression vs VADER

The Accuracy gap is:

**84.75% − 64.75% = 20.00 percentage points**

The Macro F1 gap is:

**84.74% − 63.79% = 20.95 percentage points**

This demonstrates the benefit of supervised, dataset-specific learning over a general-purpose lexicon-based approach on this benchmark.

---

# LSTM Training Behaviour

The LSTM training history shows clear signs of overfitting.

| Epoch | Training Loss | Validation Loss |
| ----: | ------------: | --------------: |
|     1 |        0.6944 |          0.6838 |
|     2 |        0.6658 |          0.6726 |
|     3 |        0.5937 |          0.6732 |
|     4 |        0.4911 |          0.7297 |
|     5 |        0.3777 |          0.7334 |

Training loss decreases consistently across the epochs, while validation loss improves initially and then increases substantially.

This indicates that the model continues fitting the training data without corresponding improvement in generalization performance.

### Interpretation

The observed behaviour suggests overfitting under the current LSTM configuration.

Potential improvements include:

* Larger training datasets
* Stronger regularization
* Systematic hyperparameter tuning
* Pretrained word embeddings
* Better sequence-length selection
* Learning-rate optimization
* Bidirectional LSTM
* Transformer-based models

---

# Computational Cost

The benchmark shows a substantial difference in computational requirements.

### Latest measured times

| Model               | Preparation + Training |
| ------------------- | ---------------------: |
| Logistic Regression |           **2.8864 s** |
| LSTM                |         **531.1268 s** |
| VADER               |                **0 s** |

The LSTM required approximately:

**531.1268 / 2.8864 ≈ 184×**

the preparation + training time of Logistic Regression.

This is an important model-engineering consideration.

A more complex architecture should provide a measurable benefit that justifies its additional computational cost. In the current experiment, the LSTM incurred substantially higher training cost while achieving lower predictive performance.

---

# Error Analysis

Detailed error analysis was performed for the Logistic Regression model.

### Test-set performance

The model produced:

* Correct predictions: **339 / 400**
* Misclassified reviews: **61 / 400**
* Error rate: **15.25%**

### Error distribution by true sentiment

| True Sentiment | Misclassified Reviews |
| -------------- | --------------------: |
| Negative       |                    35 |
| Positive       |                    26 |
| **Total**      |                **61** |

The slightly higher number of errors among negative reviews suggests that some negative examples are harder for the lexical classifier to distinguish correctly.

Difficult sentiment examples may contain mixed sentiment, subtle wording, contextual cues, sarcasm, or lexical patterns associated with the opposite sentiment class.

---

# Evaluation Strategy

The project evaluates both predictive quality and system performance.

### Classification metrics

* Accuracy
* Macro F1
* Positive-Class F1
* Precision
* Recall
* Classification report
* Confusion matrix
* ROC-AUC where probability outputs are available
* Precision-Recall analysis where probability outputs are available

### System-performance metrics

* Preparation time
* Training time
* Preparation + training time
* Test inference time
* Inference time per review

This provides a more complete basis for selecting a model than accuracy alone.

---

# Visualizations

Benchmark figures are stored under:

```text
results/
└── figures/
```

Generated visualizations include:

```text
accuracy_comparison.png
f1_comparison.png
training_time_comparison.png
inference_latency.png
lstm_training_validation_loss.png
```

These figures provide visual comparisons of predictive performance, computational cost, latency, and LSTM training behaviour.

---

# Production Recommendation

## Recommended model: TF-IDF + Logistic Regression

For the current dataset and benchmark conditions, **TF-IDF + Logistic Regression is the recommended production baseline**.

It provides the strongest overall combination of:

* Predictive performance
* Low training cost
* Low inference latency
* Simple architecture
* Easy maintenance
* Fast experimentation
* Straightforward deployment

The current benchmark achieved:

* **84.75% Accuracy**
* **84.74% Macro F1**
* **85.09% Positive-Class F1**
* **2.8864 seconds** preparation + training
* **0.1661 seconds** test-set inference
* **0.415 ms** inference per review

---

## Role of VADER

VADER remains useful as a lightweight, training-free baseline.

It can be useful when:

* labelled training data is unavailable
* rapid prototyping is required
* a simple baseline is needed
* computational resources are limited

However, VADER does not match the performance of supervised Logistic Regression on this benchmark.

---

## Role of LSTM

The LSTM provides a useful deep-learning comparison and demonstrates sequence-based sentiment modelling.

However, the current results do not support selecting it for production.

Its current computational cost is substantially higher than Logistic Regression, while its predictive performance is lower on the benchmark.

The LSTM is therefore best treated as an experimental foundation for future work involving larger datasets, pretrained embeddings, stronger regularization, systematic tuning, or more advanced sequence architectures.

---

# Project Structure

```text
nlp-sentiment-analysis-benchmark/
│
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── notebooks/
│   └── NLP-sentiment_analysis_benchmark.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── lstm_model.py
│   ├── evaluation.py
│   ├── traditional_ml.py
│   ├── utils.py
│   └── vader.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_evaluation.py
│   └── test_inference.py
│
├── models/
│   └── .gitkeep
│
└── results/
    ├── metrics.csv
    ├── comparison.csv
    └── figures/
        ├── accuracy_comparison.png
        ├── f1_comparison.png
        ├── training_time_comparison.png
        ├── inference_latency.png
        └── lstm_training_validation_loss.png
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/anisha-das-kts/nlp-sentiment-analysis-benchmark.git
cd nlp-sentiment-analysis-benchmark
```

## 2. Create a virtual environment

For Python 3.12:

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

Install the project in editable mode:

```powershell
python -m pip install -e .
```

---

# Running the Training Pipeline

Train the benchmark models with:

```powershell
python scripts/train.py
```

This produces training-time measurements under:

```text
results/metrics.csv
```

---

# Running Evaluation

Run:

```powershell
python scripts/evaluate.py
```

This produces:

```text
results/comparison.csv
```

and saves benchmark figures to:

```text
results/figures/
```

---

# Running Single-Text Inference

Example:

```powershell
python scripts/predict.py --text "I absolutely loved this movie!"
```

The script returns the predicted sentiment and the model's confidence score.

---

# Running Tests

Run:

```powershell
pytest -q
```

The repository includes tests covering preprocessing, evaluation and inference behaviour.

---

# Running the Notebook

Open:

```text
notebooks/NLP-sentiment_analysis_benchmark.ipynb
```

in VS Code and select the project's `.venv` Python environment.

The notebook walks through:

```text
Dataset Loading
      ↓
Preprocessing
      ↓
Train/Test Split
      ↓
TF-IDF + Logistic Regression
      ↓
PyTorch LSTM
      ↓
VADER
      ↓
Evaluation
      ↓
Cross-Model Benchmark
      ↓
Error Analysis
      ↓
Production Recommendation
```

---

# Reproducibility

The project uses fixed random seeds where applicable.

The train/test split is stratified, and the LSTM uses a validation split for model selection and early stopping.

Model-training and inference timings are hardware- and environment-dependent. Therefore, timing values should be interpreted as measurements from the reported execution environment rather than universal performance guarantees.

---

# Limitations

The current benchmark has several limitations:

* The dataset contains only 2,000 reviews.
* The LSTM is trained from scratch rather than using pretrained embeddings.
* The task is limited to binary sentiment classification.
* Timing measurements depend on the execution environment.
* No transformer-based model is currently included.
* Extensive hyperparameter optimization has not been performed.
* Error analysis is primarily focused on Logistic Regression.

---

# Future Improvements

Potential extensions include:

* Hyperparameter tuning with cross-validation
* Larger sentiment datasets
* Pretrained word embeddings
* Bidirectional LSTM
* Attention mechanisms
* BERT and other transformer-based models
* Probability calibration
* Model versioning
* FastAPI inference service
* Docker deployment
* Latency monitoring
* Data-drift detection
* Model-performance monitoring
* Automated retraining workflows
* CI/CD integration

---

# Conclusion

This project benchmarks three distinct approaches to sentiment analysis and demonstrates the practical trade-offs between model complexity, predictive performance and computational cost.

On the current NLTK Movie Reviews benchmark:

* **TF-IDF + Logistic Regression** achieved the best predictive performance.
* **VADER** provided a useful training-free baseline.
* **PyTorch LSTM** demonstrated deep sequence modelling but required substantially more computational effort while achieving lower performance under the current configuration.

The results support **TF-IDF + Logistic Regression as the initial production model** for this benchmark.

The broader engineering lesson is that model selection should be based on measured performance and deployment constraints rather than architectural complexity alone.

---

# License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

# Author

**Anisha Das**

Master's Student — Data Science & AI

GitHub: https://github.com/anisha-das-kts
