# Hybrid BERT Fake News Detection

A Streamlit demo for binary fake-news text classification using a hybrid feature-extraction approach. The system combines a frozen `bert-base-uncased` BERT `[CLS]` embedding (768 dimensions) with 19 linguistic features, then uses Logistic Regression for classification. SHAP provides explanations of the classifier's output.

**Disclaimer:** This is a text classifier, not an independent fact-checking service. A model prediction does not establish whether a news article is true or false.

## Installation and Usage

Requirements:

* Python 3.10 or newer
* Network access for downloading the spaCy English model and Hugging Face BERT files on first use

Create and activate a virtual environment, install dependencies, download the spaCy model, and launch the application:

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m spacy download en_core_web_sm

streamlit run app.py
```

The application loads the saved classifier and the checked-in `data/processed/shap_background.npy` file. Linguistic features and BERT embeddings are extracted from submitted titles and article text at runtime. BERT uses a maximum sequence length of 256 tokens.

CPU inference is supported, although it may be slow. Performance and package compatibility depend on the local environment.

## Dataset and Training

Training and model comparison are documented in:

`notebooks/modeling_comparison.ipynb`

The notebook expects the Kaggle Fake and Real News datasets as:

* `data/raw/Fake.csv`
* `data/raw/True.csv`

The CSV files must contain `title` and `text` columns.

**The source datasets and full cleaned dataset are not included in this repository.** Obtain the datasets from an authorized source, review their licensing terms, and do not commit the raw data.

From the repository root, prepare and clean the combined dataset:

```powershell
python src/prepare_dataset.py

python src/clean_dataset.py
```

The cleaned dataset is expected at:

`data/processed/news_clean.csv`

### Training workflow

The documented notebook workflow:

1. Loads and processes the cleaned dataset.
2. Maps the preparation labels (`0 = Fake`, `1 = Real`) to model labels (`0 = Real`, `1 = Fake`).
3. Selects 3,750 samples from each class.
4. Applies a seeded, stratified 80/20 train-test split.
5. Extracts BERT embeddings and 19 linguistic features separately.
6. Concatenates the features in this order:

   * 768 BERT `[CLS]` embedding values
   * 19 linguistic feature values
7. Trains and evaluates the classification models.

The final Logistic Regression classifier expects **787 input features**.

### Evaluation results

The project reports **99.2% accuracy** for the saved Logistic Regression model.

During the project audit, this result was recomputed using the saved model and held-out feature arrays. However, the complete training workflow was not rerun as part of that audit.

Therefore, these results should be treated as evaluation results from saved artifacts, not as an independent reproduction of the training process.

To reproduce training and evaluation, obtain the authorized source datasets, install the required dependencies, and run the notebook from beginning to end.

### External evaluation

The separate external evaluator loads the saved, unscaled Logistic Regression model and does not train or tune it. The IFND-format external dataset is intentionally excluded from version control.

#### External evaluation results

The saved classifier was evaluated on **56,714 external records**. The previously reported held-out accuracy was **99.2%**; accuracy on this external set was **70.10%**. The difference illustrates that a single held-out result does not guarantee performance on unfamiliar data.

| Metric | Result |
|---|---:|
| Accuracy | 70.10% |
| Macro-F1 | 69.57% |
| Weighted-F1 | 70.90% |
| Fake precision | 53.22% |
| Fake recall | 85.43% |
| Fake F1 | 65.58% |
| ROC-AUC | 83.16% |
| PR-AUC (average precision) | 72.12% |

Confusion matrix (rows are actual labels; columns are predicted labels):

| Actual \ Predicted | Real | Fake |
|---|---:|---:|
| Real | 23,597 | 14,203 |
| Fake | 2,756 | 16,158 |

The external data contains short statements rather than full-length articles. Statement length, publisher distribution, class balance, and dataset composition differ from training; therefore, these scores are not a like-for-like comparison with the held-out article score.

#### Publisher and statement-length analysis

Exploratory analysis found a strong publisher-label association (**Cramér's V = 0.969**). Of 29 publishers, five contained both Real and Fake examples, while 24 contained only one label. This suggests potential publisher and dataset-construction bias, but does not establish causation.

| Statement length | Records | Accuracy | Fake F1 |
|---|---:|---:|---:|
| 0–50 characters | 4,449 | 62.22% | 32.52% |
| 51–100 characters | 45,524 | 69.20% | 62.27% |
| 101–250 characters | 6,711 | 81.30% | 86.89% |
| More than 250 characters | 30 | 90.00% | 94.34% |

The final length group contains only 30 records. These bucket results are descriptive: class balance and publisher mix also vary by length, so the analysis does not isolate a causal effect of length. Detailed publisher and length tables are in `results/external_evaluation/bias_analysis/`.

The external dataset's provenance and complete independence from the training data have not been conclusively established. Exact normalized overlap checks found no matches against the available training corpus, but near-duplicates or related content cannot be ruled out.

### Reproducing External Evaluation

To reproduce the evaluation:

1. Obtain the IFND dataset from its original, legitimate source and verify its provenance and applicable usage terms.
2. Place the CSV file at `data/external/IFND.csv`.
3. Activate the project's Python virtual environment and install the dependencies listed in `requirements.txt`, if necessary.
4. Run the evaluation script from the project root:

```bash
python scripts/evaluate_external.py data/external/IFND.csv --overwrite
```

The CSV must contain `Statement` and `Label` columns. `TRUE` is mapped to Real (`0`) and `Fake` to Fake (`1`); only `Statement` is used as model input. The evaluator reports invalid/empty records, exact and approximate within-dataset duplicates, and exact normalized matches against the available training corpus. Duplicate or overlapping rows are reported, not silently filtered. Because statement-only claims are much shorter than the training articles, external results are not directly comparable to the held-out article test score.

Metrics, a text classification report, and per-record predictions are written to `results/external_evaluation/`. The command's `--overwrite` option replaces prior evaluation output files. A fresh clone requires the dataset to be obtained and placed locally before the command can run.

## Project Structure

```text
hybrid-bert-fake-news-detection/
├── app.py
├── requirements.txt
├── README.md
├── AUDIT_REPORT.md
├── src/
│   ├── prepare_dataset.py
│   ├── clean_dataset.py
│   ├── inspect_dataset.py
│   └── verify_dataset.py
├── notebooks/
│   └── modeling_comparison.ipynb
├── data/
│   └── processed/
│       ├── shap_background.npy
│       └── saved model and evaluation artifacts
└── results/
    └── figures and CSV summaries
```

The repository excludes raw datasets, the full cleaned dataset, and intermediate NumPy arrays. A compact SHAP background sample is included for the application's explanation workflow.

Dataset utility scripts resolve paths relative to the repository root, so they can be run from other working directories.

## Technologies Used

* Python
* Hugging Face Transformers
* BERT (`bert-base-uncased`)
* spaCy
* scikit-learn
* Logistic Regression
* SHAP
* Streamlit
* NumPy and Pandas

## Limitations

* Predictions reflect patterns learned from the training dataset and do not independently verify claims.
* Performance on unfamiliar publishers, topics, or time periods has not been established by the saved-model evaluation alone.
* Training reproducibility depends on dataset availability, dependency compatibility, and successful execution of the complete notebook.
* CPU inference may be slow because BERT embeddings must be computed at runtime.

## License and Dataset Attribution

Review the licenses of the source datasets and dependencies before redistributing the project or using it commercially. Add appropriate dataset attribution and a repository license once their terms have been verified.
