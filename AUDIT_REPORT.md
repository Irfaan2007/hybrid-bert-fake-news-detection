# End-to-End Project Audit

Audit date: 2026-10-26

## Audit findings

### PASS

- The saved Logistic Regression has classes `[0, 1]` and expects 787 inputs. Its evaluated input arrays have shapes `(6000, 787)` and `(1500, 787)` and all values are finite.
- The notebook fuses 768 `bert-base-uncased` `[CLS]` values followed by 19 linguistic values. The linguistic column order matches the Streamlit implementation.
- Labels are remapped consistently in the notebook and app: `0 = Real`, `1 = Fake`. The raw preparation script's `0 = Fake`, `1 = Real` is explicitly inverted before training.
- The notebook uses a fixed random seed, stratified 80/20 split, and extracts BERT and linguistic features after splitting. The scaler is fitted on training features only.
- Recomputed from the locally present saved model and held-out feature/label arrays: accuracy `0.992`, precision `0.9959677419`, recall `0.988`, F1 `0.9919678715`, ROC-AUC `0.9981208889`; confusion matrix `[[747, 3], [9, 741]]`. This validates the saved-array evaluation only; it is not a fresh training reproduction.
- The notebook's duplicate audit reports 15 repeated normalized article bodies in the cleaned data, and zero normalized body overlap in its reconstructed balanced train/test split. Local data inspection found zero conflicting labels among those repeated bodies.
- `.gitignore` excludes raw datasets, the cleaned dataset, and intermediate NumPy feature arrays. No credentials were found in the tracked file inventory.

### FAIL

- The initial app referenced ignored `X_train_fused.npy` for SHAP. Fixed by shipping a 100-row `shap_background.npy` sample and changing the app to load it. The sample is about 315 KB; full train/test arrays remain ignored.
- The initial dependency list omitted notebook imports `nltk`, `tqdm`, and `seaborn`, and there were no install/run or spaCy model setup instructions. Fixed by updating `requirements.txt` and `README.md`.
- The app initially displayed 99.20% without identifying its basis. Fixed to call it the saved held-out accuracy and clarify that the app score is not a fresh training run.
- Git branch creation (`audit/end-to-end-review`) was attempted but denied because `.git` is read-only in this environment. The working branch remains `main`.

### NEEDS INVESTIGATION

- The 99.2% score is numerically reproduced from the local saved artifacts but cannot be independently reproduced from a fresh checkout: training/test arrays and source datasets are ignored/untracked, and the notebook's source data is a locally supplied Fake/True CSV pair. Do not present this as independently validated generalization.
- The dataset has duplicate article bodies (15 groups in the saved notebook audit). The recorded split audit says none cross the split, but the source dataset's labeling conventions, provenance, and source/period confounding need manual academic review.
- A fresh package installation was not performed: this environment has an existing virtual environment, and network access is blocked. `README.md` now documents the install steps and external spaCy/Hugging Face downloads, but a clean-environment install still needs a manual run.
- Notebook output cells include local Windows paths and sample news text. They are saved outputs rather than executable hardcoded paths; clear or refresh them before academic publication if licensing or privacy review requires it.
- The repository also tracks `random_forest_model.pkl`, `logistic_regression_scaled_model.pkl`, and `standard_scaler.pkl`, which the Streamlit app does not load. The random forest artifact is about 9 MB; retain only if needed for the model comparison workflow.
- `requirements.txt` is not version-locked. A lock file and a clean install are needed to make dependency resolution repeatable across machines.

## Confirmed architecture and data-flow notes

- Feature order is BERT `[CLS]` vector (768) then linguistic values in this order: `word_count`, `sentence_count`, `character_count`, `average_word_length`, `average_sentence_length`, `noun_count`, `verb_count`, `adjective_count`, `adverb_count`, `pronoun_count`, `proper_noun_count`, `exclamation_count`, `question_count`, `uppercase_count`, `url_count`, `hashtag_count`, `mention_count`, `special_character_count`, `repeated_word_count`.
- Input construction in the app matches the notebook: `(title + " " + article).strip()`. BERT truncates at 256 tokens in both locations.
- The notebook reports both unscaled and standardized Logistic Regression results. The app loads the unscaled model, which is the model associated with the 99.2% saved evaluation. The standardized model reports 99.0% and should not be substituted without changing the reported metrics.

## Fixes and validation

- `README.md`: added Python setup, package installation, spaCy model download, app launch, input dataset requirements, label mapping, split, data exclusion, and metric caveats.
- `requirements.txt`: included direct runtime/notebook dependencies omitted from the original list (`seaborn`, `nltk`, `tqdm`, `joblib`, `jupyter`).
- `app.py`: uses the shipped SHAP background sample, accurately labels the saved held-out metric, and uses the current Streamlit dataframe width argument.
- `.gitignore`: keeps intermediate NumPy arrays excluded while allowing only `shap_background.npy` to be tracked.
- `src/prepare_dataset.py`, `src/clean_dataset.py`, `src/inspect_dataset.py`, `src/verify_dataset.py`: resolve data paths relative to the repository, independent of the caller's current directory. Preparation and cleaning now fail clearly on missing columns or non-binary source labels.
- Streamlit `AppTest` passed startup, empty input (shows expected warning), title-only, article-only, and combined input. All three non-empty submissions completed without app errors or explanation warnings. A combined sample predicted `FAKE NEWS PATTERN DETECTED` with Fake `99.63%` and Real `0.37%`; the scores sum to 100%.
- The same saved-array metric evaluation was run directly and matches the numbers above. Model load, spaCy, BERT, and SHAP all worked in the existing local environment, using cached external model files.
- Python AST parsing passed for `app.py` and all four source scripts. All notebook code cells passed AST parsing after removing IPython magic/shell lines.
- No repository test suite is present. No fresh package installation or full notebook retraining was run.
