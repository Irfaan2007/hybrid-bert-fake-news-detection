# External evaluation: publisher and text-length analysis

## Method

- Dataset: `data/external/IFND.csv`; **56,714** records.
- Existing per-record predictions were aligned to external records by unique `id` / `record_id`; all IDs matched, and all actual labels agreed.
- No BERT inference was rerun; no model, threshold, or classifier was tuned. Publisher metadata was not used as a model input.
- Length is the character count of `Statement`; buckets are 0-50, 51-100, 101-250, and >250 characters.
- Precision, recall, and F1 are Fake-positive binary metrics; undefined denominators are zero. Confusion matrices use rows=actual (Real, Fake), columns=predicted (Real, Fake).

## Publisher-label distribution

- Distinct publisher values including missing: **29**; missing publisher records: **0**.
- Publishers with both labels: **5**; single-label publishers: **24** (14 Real-only, 10 Fake-only).
- Both-label publishers: `THEPRINT`, `NEWSMOBILE`, `INDIATODAY`, `NEWSMETER`, `TEEKHIMIRCHI`.
- Real-only publishers: `TRIBUNEINDIA`, `THESTATESMAN`, `THESTATEMAN`, `INDIANEXPRESS`, `CNN`, `BSMEDIA`, `TIMESNOW`, `FACTLY`, `DAPAAN`, `DNAINDIA`, `WAHSARKAR`, `ENGLISHTRIBUNE`, `INDUSTANTIMES`, `NDTV`.
- Fake-only publishers: `AUGMENT`, `BOOMLIVE`, `ONEINDIA`, `FACTCRESCENDO`, `AFP`, `THELOGICALINDIAN`, `DIGITEYE`, `THELOGICALINDIA`, `FACTCHECKER`, `ALT`.
- Publisher-label association: Pearson chi-square(28) = **53249.79**, p **< 0.001**, Cramer's V = **0.969**. The calculated p-value underflowed numerically to zero; this describes association, not causation.
- Per-publisher counts, label shares, majority share, imbalance ratio, and scores are in `publisher_label_distribution.csv`.

### Largest publishers by record count

| Publisher | Records | Real | Fake | Fake share | Labels |
|---|---:|---:|---:|---:|---|
| TRIBUNEINDIA | 11,832 | 11,832 | 0 | 0.0% | Real only |
| AUGMENT | 11,321 | 0 | 11,321 | 100.0% | Fake only |
| THEPRINT | 8,582 | 8,238 | 344 | 4.0% | both |
| THESTATESMAN | 7,202 | 7,202 | 0 | 0.0% | Real only |
| THESTATEMAN | 4,048 | 4,048 | 0 | 0.0% | Real only |
| INDIANEXPRESS | 3,983 | 3,983 | 0 | 0.0% | Real only |
| NEWSMOBILE | 2,422 | 232 | 2,190 | 90.4% | both |
| INDIATODAY | 1,698 | 92 | 1,606 | 94.6% | both |
| BOOMLIVE | 806 | 0 | 806 | 100.0% | Fake only |
| CNN | 778 | 778 | 0 | 0.0% | Real only |
| NEWSMETER | 572 | 66 | 506 | 88.5% | both |
| ONEINDIA | 549 | 0 | 549 | 100.0% | Fake only |

### Performance by publisher class support

| Publisher cohort | Publishers | Records | Real | Fake | Accuracy | Fake precision | Fake recall | Fake F1 | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Publishers with both labels | 5 | 13,661 | 8,753 | 4,908 | 0.6942 | 0.5441 | 0.9181 | 0.6833 | `[[4978, 3775], [402, 4506]]` |
| Publishers with one label only | 24 | 43,053 | 29,047 | 14,006 | 0.7031 | 0.5277 | 0.8319 | 0.6458 | `[[18619, 10428], [2354, 11652]]` |

Pooled scores for publishers containing only one observed class are composition-dependent. Although the combined cohort may contain both labels across its publishers, a single publisher cannot support within-publisher comparison on an absent class; such metrics may mislead.

## Performance by statement length

| Length bucket | N | Real | Fake | Accuracy | Fake precision | Fake recall | Fake F1 | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0-50 characters | 4,449 | 3,921 | 528 | 0.6222 | 0.2063 | 0.7670 | 0.3252 | `[[2363, 1558], [123, 405]]` |
| 51-100 characters | 45,524 | 31,781 | 13,743 | 0.6920 | 0.4941 | 0.8418 | 0.6227 | `[[19935, 11846], [2174, 11569]]` |
| 101-250 characters | 6,711 | 2,095 | 4,616 | 0.8130 | 0.8390 | 0.9010 | 0.8689 | `[[1297, 798], [457, 4159]]` |
| More than 250 characters | 30 | 3 | 27 | 0.9000 | 0.9615 | 0.9259 | 0.9434 | `[[2, 1], [2, 25]]` |

## Interpretation and limitations

- Publisher-label association and observed score differences can be consistent with publisher/source bias, but do not establish that the source caused prediction errors.
- Accuracy rises descriptively across the first three length buckets (0.622, 0.692, 0.813), while the >250-character bucket has only 30 records. Length appears associated with performance in this dataset, but this is not a causal estimate.
- Label distribution and publisher composition can differ by length, so the buckets do not isolate a causal length effect.
- Performance on publishers with only one label is not a test of discrimination within that publisher.
- The external set consists of short statements while training documents are much longer articles, limiting direct comparability.

Detailed publisher and bucket results are in the accompanying CSV files.
