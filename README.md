# NLP Sentiment Analysis Benchmark

A reproducible NLP benchmarking project comparing three sentiment-analysis approaches on the NLTK Movie Reviews dataset:

1. **TF-IDF + Logistic Regression** — traditional supervised machine learning
2. **PyTorch LSTM** — sequence-based deep learning with learned word embeddings
3. **VADER** — training-free lexicon/rule-based sentiment analysis

The project evaluates not only predictive quality, but also computational cost, inference latency, class-level behaviour, uncertainty, and practical deployment trade-offs.

## Project Highlights

- 2,000 labelled movie reviews: 1,000 positive and 1,000 negative
- Stratified 80/20 train-test split: 1,600 training and 400 test reviews
- Negation-aware text preprocessing
- TF-IDF unigram/bigram representation
- Logistic Regression baseline
- PyTorch LSTM with a controlled vocabulary, padding, validation split, and early stopping
- VADER training-free baseline
- Accuracy, Macro F1, Positive-Class F1, classification reports, and confusion matrices
- Training/preparation time and test inference latency measurements
- Logistic Regression error, uncertainty, and confident-error analysis
- Benchmark visualizations under `results/figures/`

## Dataset

The project uses the **NLTK Movie Reviews** corpus. The dataset is downloaded programmatically through NLTK, so the corpus itself is not committed to the repository.

| Split | Reviews | Positive | Negative |
|---|---:|---:|---:|
| Full dataset | 2,000 | 1,000 | 1,000 |
| Training | 1,600 | 800 | 800 |
| Test | 400 | 200 | 200 |

The train/test split is stratified with random seed `42`.

## Preprocessing

The text pipeline performs:

- lowercasing
- contracted-negation normalization
- URL and HTML removal
- punctuation and numeric removal
- tokenization
- task-aware stop-word removal while preserving `no`, `nor`, and `not`
- one-character token removal
- whitespace normalization
- safe duplicate handling when duplicate reviews have consistent labels

The vocabulary for the LSTM is built from training text only to avoid test-set leakage.

## Model Architecture

### TF-IDF + Logistic Regression

Cleaned reviews are converted into sparse TF-IDF features using unigrams and bigrams, followed by Logistic Regression for binary classification.

### PyTorch LSTM

The LSTM builds a vocabulary from training reviews, converts text to integer sequences, truncates/pads sequences, applies a learned embedding layer, and feeds the sequence representation into an LSTM followed by dropout and a linear classifier.

Validation loss is monitored during training and the best validation checkpoint is restored.

### VADER

VADER is evaluated without fitting on the movie-review training set. It uses a predefined sentiment lexicon and rule-based scoring as a training-free baseline.

## Benchmark Results

The table below reflects the latest completed local benchmark snapshot used for the portfolio project. Runtime measurements are environment-dependent and should be treated as observed benchmark values, not universal guarantees.

| Model | Accuracy | Macro F1 | Positive F1 | Prep + Training | Test Inference | Inference / Review |
|---|---:|---:|---:|---:|---:|---:|
| **TF-IDF + Logistic Regression** | **84.75%** | **84.74%** | **85.09%** | **2.9499 s** | **0.3840 s** | **0.9600 ms** |
| PyTorch LSTM | 52.75% | 52.62% | 50.13% | 687.7271 s | 1.7683 s | 4.4208 ms |
| VADER | 64.75% | 63.79% | 69.68% | 0.0000 s | 2.5290 s | 6.3225 ms |

### Key Findings

- **TF-IDF + Logistic Regression is the strongest model** on Accuracy and Macro F1 in this experiment.
- Logistic Regression leads LSTM by **32.00 percentage points** in Accuracy.
- Logistic Regression leads VADER by **20.00 percentage points** in Accuracy.
- LSTM required approximately **233×** the preparation + training time of Logistic Regression in this benchmark.
- Logistic Regression also had the lowest measured test-set inference time per review.
- The LSTM training history shows validation loss worsening after the best validation point while training loss continues to fall, consistent with overfitting behaviour in this run.

## LSTM Training Behaviour

The latest training history was:

| Epoch | Training Loss | Validation Loss |
|---:|---:|---:|
| 1 | 0.6944 | 0.6838 |
| 2 | 0.6658 | 0.6726 |
| 3 | 0.5937 | 0.6732 |
| 4 | 0.4911 | 0.7297 |
| 5 | 0.3777 | 0.7334 |

The widening separation after epoch 2 indicates that the model continued fitting the training data while validation performance deteriorated.

## Error and Uncertainty Analysis

For Logistic Regression, the notebook examines:

- total misclassifications
- error distribution by true sentiment
- predictions near the decision boundary
- confidently misclassified examples
- probability-based uncertainty

On the 400-review test set, Logistic Regression misclassified **61 reviews**: 35 negative reviews and 26 positive reviews.

## Production Recommendation

For the current dataset and benchmark conditions, **TF-IDF + Logistic Regression is the recommended initial production model**.

It combines the best predictive performance in this experiment with low training cost, low inference latency, simple implementation, and straightforward maintenance.

VADER remains useful when labelled training data is unavailable and a training-free baseline is preferred. The LSTM remains an experimental candidate for future work with larger datasets, pretrained embeddings, stronger regularization, or more advanced contextual models.

## Visualizations

Generated figures are stored in `results/figures/`:

- `accuracy_comparison.png`
- `f1_comparison.png`
- `training_time_comparison.png`
- `inference_latency.png`
- `lstm_training_validation_loss.png`

## Project Repository Structure

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

## Setup

### 1. Clone

```bash
git clone https://github.com/anisha-das-kts/nlp-sentiment-analysis-benchmark.git
cd nlp-sentiment-analysis-benchmark
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For editable installation:

```powershell
pip install -e .
```

### 4. Run tests

```powershell
pytest -q
```

### 5. Train the models

```powershell
python scripts/train.py
```

### 6. Evaluate the benchmark

```powershell
python scripts/evaluate.py
```

### 7. Run an example prediction

```powershell
python scripts/predict.py "I absolutely loved this movie!"
```

### 8. Open the notebook

```powershell
jupyter notebook notebooks/sentiment_analysis_benchmark.ipynb
```

The notebook downloads the required NLTK resources automatically.

## Reproducibility

The project uses random seed `42` across Python, NumPy, and PyTorch. Dataset splitting is stratified, and the LSTM vocabulary is learned from training data only.

Timing values should be regenerated when comparing different machines or environments. For a benchmark snapshot, run the complete pipeline from a clean state and keep the README, notebook, and CSV results synchronized.

## Limitations

- The dataset contains only 2,000 labelled reviews.
- Runtime measurements are local and environment-dependent.
- VADER is a general-purpose rule/lexicon baseline rather than a movie-review-specific model.
- The LSTM is trained from scratch rather than using pretrained contextual embeddings.
- The benchmark does not provide production-scale concurrent load testing.

## Future Improvements

- Tune TF-IDF and Logistic Regression hyperparameters systematically
- Improve the LSTM with pretrained embeddings and stronger regularization
- Add transformer-based sentiment models
- Add probability calibration
- Add a lightweight FastAPI inference service
- Add model persistence/versioning workflows
- Add richer monitoring and drift analysis

## License

MIT License. See `LICENSE` for details.

## Author

Anisha Das
