#!/usr/bin/env python
"""Evaluate the saved hybrid BERT classifier on a labeled external CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import spacy
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "logistic_regression_model.pkl"
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "news_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "external_evaluation"

BERT_MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
LINGUISTIC_FEATURE_NAMES = [
    "word_count",
    "sentence_count",
    "character_count",
    "average_word_length",
    "average_sentence_length",
    "noun_count",
    "verb_count",
    "adjective_count",
    "adverb_count",
    "pronoun_count",
    "proper_noun_count",
    "exclamation_count",
    "question_count",
    "uppercase_count",
    "url_count",
    "hashtag_count",
    "mention_count",
    "special_character_count",
    "repeated_word_count",
]


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def extract_linguistic_features(doc: Any) -> np.ndarray:
    """Keep feature definitions and ordering aligned with app.py."""
    text = doc.text
    noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")
    adjective_count = sum(1 for token in doc if token.pos_ == "ADJ")
    adverb_count = sum(1 for token in doc if token.pos_ == "ADV")
    pronoun_count = sum(1 for token in doc if token.pos_ == "PRON")
    proper_noun_count = sum(1 for token in doc if token.pos_ == "PROPN")

    words = [
        token.text
        for token in doc
        if not token.is_punct and not token.is_space
    ]
    word_count = len(words)
    sentence_count = len(list(doc.sents))
    character_count = len(text)
    average_word_length = (
        sum(len(word) for word in words) / word_count if word_count else 0
    )
    average_sentence_length = (
        word_count / sentence_count if sentence_count else 0
    )
    exclamation_count = text.count("!")
    question_count = text.count("?")
    uppercase_count = sum(
        1 for word in words if word.isupper() and len(word) > 1
    )
    url_count = len(re.findall(r"https?://\S+|www\.\S+", text))
    hashtag_count = len(re.findall(r"#\w+", text))
    mention_count = len(re.findall(r"@\w+", text))
    special_character_count = len(
        re.findall(r"[^a-zA-Z0-9\s]", text)
    )
    word_counter = Counter(word.lower() for word in words)
    repeated_word_count = sum(1 for count in word_counter.values() if count > 2)

    features = np.asarray(
        [
            word_count,
            sentence_count,
            character_count,
            average_word_length,
            average_sentence_length,
            noun_count,
            verb_count,
            adjective_count,
            adverb_count,
            pronoun_count,
            proper_noun_count,
            exclamation_count,
            question_count,
            uppercase_count,
            url_count,
            hashtag_count,
            mention_count,
            special_character_count,
            repeated_word_count,
        ],
        dtype=np.float64,
    )
    if features.shape != (19,) or not np.isfinite(features).all():
        raise ValueError(f"Invalid linguistic feature vector: {features.shape}")
    return features


def get_column_name(frame: pd.DataFrame, expected: str) -> str | None:
    return next(
        (column for column in frame.columns if column.casefold() == expected.casefold()),
        None,
    )


def get_text_lookups(training_path: Path) -> tuple[dict[str, set[str]], int]:
    if not training_path.is_file():
        raise FileNotFoundError(
            "Cannot verify training-corpus overlap; expected file is missing: "
            f"{training_path}"
        )

    lookups = {"title": set(), "text": set(), "title_and_text": set()}
    training_rows = 0
    with training_path.open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as source:
        reader = csv.DictReader(source)
        if not {"title", "text"}.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Training data must include title and text columns: {training_path}"
            )
        for row in reader:
            training_rows += 1
            title = normalize_text(row.get("title"))
            body = normalize_text(row.get("text"))
            if title:
                lookups["title"].add(title)
            if body:
                lookups["text"].add(body)
            combined = normalize_text(
                f"{row.get('title') or ''} {row.get('text') or ''}"
            )
            if combined:
                lookups["title_and_text"].add(combined)
    return lookups, training_rows


def token_shingles(text: str) -> set[str]:
    tokens = re.findall(r"\w+", text)
    if len(tokens) < 3:
        return set(tokens)
    return {
        " ".join(tokens[index : index + 3])
        for index in range(len(tokens) - 2)
    }


def near_duplicate_statistics(
    normalized_texts: list[str], similarity_threshold: float = 0.8
) -> dict[str, Any]:
    """Find likely near-duplicates using MinHash candidate buckets."""
    unique_texts = list(dict.fromkeys(text for text in normalized_texts if text))
    shingles = [token_shingles(text) for text in unique_texts]
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    signature_size = 32
    band_size = 4

    for text_index, text_shingles in enumerate(shingles):
        if not text_shingles:
            continue
        signature = [(1 << 64) - 1] * signature_size
        encoded_shingles = [item.encode("utf-8") for item in text_shingles]
        for seed in range(signature_size):
            prefix = seed.to_bytes(2, "little")
            signature[seed] = min(
                int.from_bytes(
                    hashlib.blake2b(prefix + shingle, digest_size=8).digest(),
                    "big",
                )
                for shingle in encoded_shingles
            )
        for band in range(signature_size // band_size):
            start = band * band_size
            buckets[(band, tuple(signature[start : start + band_size]))].append(
                text_index
            )

    candidate_pairs: set[tuple[int, int]] = set()
    max_candidates = 250_000
    candidate_limit_reached = False
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for left, right in itertools.combinations(bucket, 2):
            candidate_pairs.add((left, right))
            if len(candidate_pairs) >= max_candidates:
                candidate_limit_reached = True
                break
        if candidate_limit_reached:
            break

    near_pairs: list[tuple[int, int]] = []
    for left, right in candidate_pairs:
        union_size = len(shingles[left] | shingles[right])
        if union_size and (
            len(shingles[left] & shingles[right]) / union_size
            >= similarity_threshold
        ):
            near_pairs.append((left, right))

    involved_unique = {index for pair in near_pairs for index in pair}
    multiplicities = Counter(normalized_texts)
    involved_rows = sum(multiplicities[unique_texts[index]] for index in involved_unique)
    return {
        "method": (
            "Approximate MinHash candidate search over normalized word 3-gram "
            "sets (word tokens used for texts shorter than 3 words); "
            f"Jaccard threshold {similarity_threshold:.2f}."
        ),
        "unique_nonempty_texts_checked": len(unique_texts),
        "candidate_pairs_checked": len(candidate_pairs),
        "candidate_pair_limit": max_candidates,
        "candidate_limit_reached": candidate_limit_reached,
        "near_duplicate_pairs": len(near_pairs),
        "unique_texts_in_near_duplicate_pairs": len(involved_unique),
        "rows_in_near_duplicate_pairs": involved_rows,
        "records_filtered_for_near_duplicates": 0,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the saved hybrid BERT classifier on external labeled text. "
            "External labels are used only for final metrics."
        )
    )
    parser.add_argument("csv_path", type=Path, help="Path to the external CSV file")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="spaCy/BERT inference batch size (default: 16)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace prior files in the selected output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    csv_path = args.csv_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be a positive integer")
    if not csv_path.is_file():
        raise FileNotFoundError(f"External CSV not found: {csv_path}")

    output_paths = [
        output_dir / "metrics.json",
        output_dir / "classification_report.txt",
        output_dir / "predictions.csv",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing evaluation outputs. Use --overwrite "
            "only if replacing them is intended: "
            + ", ".join(str(path) for path in existing)
        )

    try:
        frame = pd.read_csv(
            csv_path,
            dtype=object,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        source_encoding = "utf-8-sig"
    except UnicodeDecodeError:
        frame = pd.read_csv(
            csv_path,
            dtype=object,
            keep_default_na=False,
            encoding="cp1252",
            encoding_errors="replace",
        )
        source_encoding = "cp1252 (undefined bytes replaced)"
    statement_column = get_column_name(frame, "Statement")
    label_column = get_column_name(frame, "Label")
    if statement_column is None or label_column is None:
        raise ValueError(
            "External CSV must have Statement and Label columns; found "
            f"{frame.columns.tolist()}"
        )

    id_column = get_column_name(frame, "id")
    web_column = get_column_name(frame, "Web")
    category_column = get_column_name(frame, "Category")
    date_column = get_column_name(frame, "Date")

    raw_labels = frame[label_column].astype(str).str.strip()
    normalized_labels = raw_labels.str.casefold()
    observed_labels = sorted(set(raw_labels[raw_labels != ""]))
    accepted_labels = {"true", "fake"}
    unknown_label_mask = ~normalized_labels.isin(accepted_labels)
    missing_text_mask = frame[statement_column].map(
        lambda value: not normalize_text(value)
    )
    invalid_label_mask = unknown_label_mask
    excluded_mask = invalid_label_mask | missing_text_mask
    valid_mask = ~excluded_mask

    if not valid_mask.any():
        raise ValueError(
            "No evaluable rows remain after excluding empty statements and "
            "unrecognized labels."
        )
    if not set(normalized_labels[valid_mask]).issubset(accepted_labels):
        raise ValueError("Unexpected labels remained after label validation.")

    label_map = {"true": 0, "fake": 1}
    actual = normalized_labels[valid_mask].map(label_map).to_numpy(dtype=np.int64)
    texts = frame.loc[valid_mask, statement_column].map(
        lambda value: str(value).strip()
    ).tolist()
    source_rows = (frame.index[valid_mask].to_numpy(dtype=np.int64) + 2).tolist()
    valid_source = frame.loc[valid_mask].reset_index(drop=True)
    normalized_texts = [normalize_text(text) for text in texts]

    duplicate_counts = Counter(normalized_texts)
    duplicate_groups = sum(count > 1 for count in duplicate_counts.values())
    duplicate_extra_rows = sum(
        count - 1 for count in duplicate_counts.values() if count > 1
    )
    near_duplicates = near_duplicate_statistics(normalized_texts)

    training_lookups, training_rows = get_text_lookups(TRAINING_DATA_PATH)
    overlap_types: list[str] = []
    overlap_counts = Counter()
    for text in normalized_texts:
        kinds = [
            key
            for key, values in training_lookups.items()
            if text and text in values
        ]
        overlap_types.append(";".join(kinds))
        overlap_counts.update(kinds)

    print(f"External CSV: {csv_path}")
    print(f"Rows: {len(frame):,}; evaluable: {len(texts):,}")
    print(
        f"Excluded rows: {int(excluded_mask.sum()):,} "
        f"(invalid/missing label: {int(invalid_label_mask.sum()):,}; "
        f"empty statement: {int(missing_text_mask.sum()):,})"
    )
    print(f"Verified label values observed: {observed_labels}")
    print(
        f"Exact training overlap counts: {dict(overlap_counts)} "
        f"(training rows checked: {training_rows:,})"
    )
    print(
        "Loading saved classifier, spaCy, and bert-base-uncased; "
        f"device={('cuda' if torch.cuda.is_available() else 'cpu')}."
    )

    classifier = joblib.load(MODEL_PATH)
    if getattr(classifier, "n_features_in_", None) != 787:
        raise ValueError(
            "Saved classifier feature count is not 787: "
            f"{getattr(classifier, 'n_features_in_', 'unknown')}"
        )
    if list(classifier.classes_) != [0, 1]:
        raise ValueError(
            "Expected saved classifier classes [0, 1], found "
            f"{list(classifier.classes_)}"
        )

    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    bert_model.eval()

    predicted: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    total_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    for start in range(0, len(texts), args.batch_size):
        stop = min(start + args.batch_size, len(texts))
        text_batch = texts[start:stop]
        docs = list(nlp.pipe(text_batch, batch_size=args.batch_size))
        linguistic = np.vstack(
            [extract_linguistic_features(doc) for doc in docs]
        )
        tokenized = tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            outputs = bert_model(**tokenized)
        bert_features = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        if bert_features.shape != (len(text_batch), 768):
            raise ValueError(
                "Unexpected BERT feature shape: "
                f"{bert_features.shape}; expected ({len(text_batch)}, 768)"
            )
        fused = np.concatenate((bert_features, linguistic), axis=1)
        if fused.shape != (len(text_batch), 787) or not np.isfinite(fused).all():
            raise ValueError(f"Invalid fused feature batch: {fused.shape}")
        predicted.append(classifier.predict(fused).astype(np.int64))
        probabilities.append(classifier.predict_proba(fused))
        batch_number = start // args.batch_size + 1
        if batch_number == 1 or batch_number % 100 == 0 or stop == len(texts):
            print(f"Processed {stop:,}/{len(texts):,} records.")

    y_pred = np.concatenate(predicted)
    class_probabilities = np.vstack(probabilities)
    prob_real = class_probabilities[:, 0]
    prob_fake = class_probabilities[:, 1]

    confusion = confusion_matrix(actual, y_pred, labels=[0, 1])
    roc_auc: float | None = None
    pr_auc: float | None = None
    roc_auc_note: str | None = None
    if len(np.unique(actual)) == 2:
        roc_auc = float(roc_auc_score(actual, prob_fake))
        pr_auc = float(average_precision_score(actual, prob_fake))
    else:
        roc_auc_note = "Undefined: evaluation data contains only one true class."

    report_text = classification_report(
        actual,
        y_pred,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        zero_division=0,
        digits=4,
    )
    label_distribution = {
        "Real": int(np.count_nonzero(actual == 0)),
        "Fake": int(np.count_nonzero(actual == 1)),
    }
    source_bias: dict[str, Any] = {}
    if web_column:
        web_values = valid_source[web_column].astype(str).str.strip()
        top_webs = web_values.value_counts().head(15).index.tolist()
        source_bias = {
            web: {
                "records": int((web_values == web).sum()),
                "real": int(np.count_nonzero(actual[(web_values == web).to_numpy()] == 0)),
                "fake": int(np.count_nonzero(actual[(web_values == web).to_numpy()] == 1)),
            }
            for web in top_webs
        }

    statement_lengths = np.asarray([len(text) for text in texts], dtype=np.int64)
    training_length_summary: dict[str, float] | None = None
    if TRAINING_DATA_PATH.is_file():
        lengths: list[int] = []
        with TRAINING_DATA_PATH.open(
            "r", encoding="utf-8-sig", errors="replace", newline=""
        ) as source:
            for row in csv.DictReader(source):
                document = f"{row.get('title') or ''} {row.get('text') or ''}".strip()
                if document:
                    lengths.append(len(document))
        if lengths:
            training_length_summary = {
                "median_characters": float(np.median(lengths)),
                "p90_characters": float(np.percentile(lengths, 90)),
            }

    metrics: dict[str, Any] = {
        "dataset": {
            "path": str(csv_path),
            "filename_identity": csv_path.name,
            "text_encoding": source_encoding,
            "rows_total": int(len(frame)),
            "rows_evaluated": int(len(texts)),
            "rows_excluded": int(excluded_mask.sum()),
            "excluded_invalid_or_missing_label": int(invalid_label_mask.sum()),
            "excluded_empty_statement": int(missing_text_mask.sum()),
            "observed_nonempty_label_values": observed_labels,
            "label_mapping": {"TRUE": 0, "Fake": 1},
            "label_distribution": label_distribution,
            "input_text_column": statement_column,
            "metadata_excluded_from_model_input": [
                column
                for column in (id_column, web_column, category_column, date_column)
                if column
            ],
            "text_format_note": (
                "Statement-only short claims/headline-like records are evaluated "
                "without adding or fabricating article text."
            ),
            "statement_length_characters": {
                "median": float(np.median(statement_lengths)),
                "p90": float(np.percentile(statement_lengths, 90)),
                "maximum": int(statement_lengths.max()),
            },
            "training_combined_document_length_characters": training_length_summary,
        },
        "model": {
            "path": str(MODEL_PATH),
            "classifier": type(classifier).__name__,
            "bert_model": BERT_MODEL_NAME,
            "bert_representation": "last_hidden_state[:, 0, :] (CLS)",
            "bert_max_length": MAX_LENGTH,
            "bert_truncation": True,
            "linguistic_feature_names": LINGUISTIC_FEATURE_NAMES,
            "feature_order": "768 BERT CLS values, then the 19 listed linguistic values",
            "feature_count": 787,
            "classifier_classes": [int(item) for item in classifier.classes_],
            "scaling": "None; app uses the saved unscaled Logistic Regression.",
        },
        "metrics": {
            "accuracy": float(accuracy_score(actual, y_pred)),
            "precision_fake_positive": float(
                precision_score(actual, y_pred, pos_label=1, zero_division=0)
            ),
            "recall_fake_positive": float(
                recall_score(actual, y_pred, pos_label=1, zero_division=0)
            ),
            "f1_fake_positive": float(
                f1_score(actual, y_pred, pos_label=1, zero_division=0)
            ),
            "macro_f1": float(
                f1_score(actual, y_pred, labels=[0, 1], average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(
                    actual, y_pred, labels=[0, 1], average="weighted", zero_division=0
                )
            ),
            "roc_auc_fake": roc_auc,
            "precision_recall_auc_average_precision": pr_auc,
            "roc_auc_note": roc_auc_note,
            "confusion_matrix": {
                "labels_order": ["Real", "Fake"],
                "rows_actual_columns_predicted": confusion.tolist(),
            },
            "classification_report": report_text,
        },
        "data_quality": {
            "exact_duplicate_normalized_statement_groups": int(duplicate_groups),
            "exact_duplicate_extra_rows": int(duplicate_extra_rows),
            "near_duplicates": near_duplicates,
            "training_overlap": {
                "training_corpus_path": str(TRAINING_DATA_PATH),
                "training_rows_checked": training_rows,
                "normalization": (
                    "Unicode NFKC, casefold, whitespace collapse; punctuation retained."
                ),
                "exact_external_record_matches": {
                    "training_title": int(overlap_counts["title"]),
                    "training_body": int(overlap_counts["text"]),
                    "training_title_and_body": int(overlap_counts["title_and_text"]),
                    "any_training_field": int(sum(bool(item) for item in overlap_types)),
                },
                "records_removed_for_overlap": 0,
            },
            "source_web_label_counts_top_15": source_bias,
            "dataset_provenance_note": (
                "The filename and fields are consistent with an IFND-format Indian "
                "fake-news dataset, but this local archive does not authenticate "
                "the original publisher or collection provenance."
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(
        {
            "source_row": source_rows,
            "record_id": (
                valid_source[id_column].astype(str).tolist()
                if id_column
                else [""] * len(valid_source)
            ),
            "actual_label": np.where(actual == 0, "Real", "Fake"),
            "predicted_label": np.where(y_pred == 0, "Real", "Fake"),
            "probability_real": prob_real,
            "probability_fake": prob_fake,
            "exact_training_overlap_fields": overlap_types,
        }
    )
    if web_column:
        prediction_frame["source_web"] = valid_source[web_column].astype(str).tolist()
    if category_column:
        prediction_frame["source_category"] = valid_source[category_column].astype(str).tolist()
    if date_column:
        prediction_frame["source_date"] = valid_source[date_column].astype(str).tolist()

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_header = (
        "External evaluation of saved unscaled Logistic Regression\n"
        f"Dataset: {csv_path}\n"
        f"Evaluated records: {len(texts)}\n"
        "Confusion matrix (rows actual, columns predicted; Real, Fake):\n"
        f"{confusion.tolist()}\n\n"
    )
    (output_dir / "classification_report.txt").write_text(
        report_header + report_text, encoding="utf-8"
    )
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)

    print("\nExternal evaluation complete.")
    print(json.dumps(metrics["metrics"], indent=2))
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
