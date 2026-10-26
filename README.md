# Hybrid BERT Fake News Detection

Streamlit demo for a binary text classifier that combines a frozen `bert-base-uncased` `[CLS]` embedding (768 values) with 19 linguistic features, then predicts with Logistic Regression. SHAP explains the classifier output. A model score is not independent fact verification.

## Install and run

Use Python 3.10 or newer in a virtual environment. The first run downloads the spaCy English model and Hugging Face BERT files, so it needs network access.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

The app loads the checked-in classifier and `data/processed/shap_background.npy`. It extracts features from each submitted title and article at runtime. BERT uses a maximum sequence length of 256.

## Data and training

Training is documented in `notebooks/modeling_comparison.ipynb`. It expects the Kaggle Fake and Real news CSV files as `data/raw/Fake.csv` and `data/raw/True.csv`, with `title` and `text` columns. The repository does not include these source datasets or the full cleaned dataset. Obtain them from an authorized source, review their license, and do not commit them. From the repository root, prepare and clean the combined CSV before running the notebook:

```powershell
python src/prepare_dataset.py
python src/clean_dataset.py
```

The notebook expects the cleaned dataset at `data/processed/news_clean.csv`. It maps the preparation labels (`0 = Fake`, `1 = Real`) to model labels (`0 = Real`, `1 = Fake`), selects 3,750 rows per class, and applies a seeded stratified 80/20 split. BERT embeddings and linguistic features are extracted separately after that split and concatenated in the order 768 BERT values then the 19 features listed in the notebook.

The checked-in results and notebook outputs are saved project artifacts, not an independent reproduction. The reported 99.2% accuracy was recomputed from the local saved model and held-out arrays during the audit; reproducing training requires the source data, dependencies, and a complete notebook run.

## Repository contents

- `app.py`: Streamlit interface, feature extraction, Logistic Regression, and SHAP.
- `src/`: dataset combine, clean, inspect, and verify utilities.
- `notebooks/modeling_comparison.ipynb`: feature extraction, modeling, and saved evaluation workflow.
- `data/processed/`: checked-in models, metrics, and the compact SHAP background sample. Full datasets and intermediate NumPy arrays are excluded.
- `results/`: figures and CSV summaries saved by the notebook.

Run dataset utilities from any working directory; paths resolve from the repository root. Python package compatibility and GPU performance depend on the local platform. CPU inference is supported but can be slow.
