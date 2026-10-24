# Grammar Scoring Engine for Spoken English

Predict a **grammar score (0–5)** from a 45–60 second audio clip of speech.
Built for a private Kaggle prediction competition (769 labelled training clips, 216 test clips).
Scores are MOS-style Likert ratings from human raters; the leaderboard uses **Pearson correlation and RMSE**.

> **Status:** work in progress. The pipeline is built step by step; the results table below is filled in only with numbers I have actually run.

## Approach

Grammar lives in the *words*, not the sound, so the main path is speech-to-text followed by text analysis. Only the last stage is trained.

```
audio.wav ──► Whisper (pretrained, frozen) ──► transcript + word timestamps
                                                   │
        LanguageTool · GPT-2 perplexity · CoLA ────┤  (pretrained, frozen scorers)
        sentence stats · fluency (pauses, rate) ───┘
                                                   ▼
                                  table of ~18 numeric features per clip
                                                   ▼
                              LightGBM regression (5-fold CV, trained)
                                                   ▼
                                    clipped grammar score (0–5)
```

This is classic machine-learning regression on top of frozen deep-learning feature extractors. With only 769 samples, training a neural network from scratch would memorise the data.

### Features

| Group | Features |
|---|---|
| Grammar | LanguageTool errors (total, per word), GPT-2 perplexity, CoLA acceptability (mean, min over sentences) |
| Structure | word count, sentence count, average sentence length, type-token ratio, fillers, immediate repeats |
| Fluency | words per minute, articulation rate, long-pause count, mean pause, pause ratio, duration |

### Evaluation protocol

- **Out-of-fold (OOF) RMSE and Pearson r** from stratified 5-fold CV: the honest estimate.
- **Training RMSE**: required by the competition; it is always optimistic, and the gap to OOF shows overfitting.
- Metrics are reported on all 769 clips **and** on the 732 real-speech clips separately (see findings).

## Data findings so far

Computed directly from `train.csv`:

| Finding | Detail |
|---|---|
| Size | 769 rows, unique filenames, label range 0.0 to 5.0, all labels are multiples of 0.5 |
| Two populations | 37 files `audio_5037` to `audio_5073` all have label **0.0**; the other 732 files (`audio_0` to `audio_784`) are labelled 1.0 to 5.0 |
| Imbalanced scale | Score 1.0 has 1 clip and 1.5 has 3 clips, so the bottom of the scale is almost unlearnable |
| Test set | `test.csv` labels are placeholders and are ignored; only the 216 filenames matter |

Baselines every model has to beat (training-set RMSE):

| Baseline | RMSE |
|---|---|
| Predict the global mean (3.31) | 1.238 |
| `audio_50xx` → 0, everything else → 3.48 | 0.989 |
| Regular clips only, predict their mean | 1.014 |

### Open questions (not yet verified)

- Whether the `audio_50xx` clips are silence/noise (needs listening and Whisper transcripts).
- Whether Whisper "repairs" speakers' grammar mistakes and hides the signal.
- Whether test filenames that coincide with train filenames are different recordings (check file checksums).

## Results

| Model | Features | Training RMSE | OOF RMSE | OOF Pearson r |
|---|---|---|---|---|
| Mean baseline | none | 1.238 | n/a | n/a |
| LightGBM, 5-fold | text + fluency | TBD | TBD | TBD |
| DeBERTa fine-tune (planned) | transcript | TBD | TBD | TBD |
| Blend (planned) | | TBD | TBD | TBD |

## Repository layout

```
.
├── README.md
├── requirements.txt
├── data/                  # NOT committed: put the competition data here (see data/README.md)
├── notebooks/
│   └── 01_grammar_scoring_pipeline.ipynb   # main deliverable: code + report
├── docs/
│   ├── findings.md        # running log of data findings and decisions
│   └── roadmap.md         # build plan and status
└── outputs/               # figures and submission.csv (small files only)
```

## How to run

1. Open the notebook in Google Colab or Kaggle with a GPU (Whisper needs one for reasonable speed).
2. Install dependencies (the first cells of the notebook do this; Java is required by LanguageTool).
3. Set `DATA_DIR` in the setup cell to the folder holding `train/`, `test/` and the CSV files.
4. Run all cells. Transcription is checkpointed to disk after every clip, so an interrupted session can resume.

Random seeds are fixed (`SEED = 42`).

## Data

The competition audio and labels are not included in this repository (private competition data, large files). Expected layout:

```
Dataset_Final/
├── train/                 # 769 .wav files
├── test/                  # 216 .wav files
├── train.csv              # filename, label
├── test.csv               # filename, label (placeholder values, ignored)
└── sample_submission.csv
```

## Roadmap

- [x] Data audit and baselines
- [ ] Whisper transcription of train and test, with a manual quality check
- [ ] Text and fluency features, LightGBM with 5-fold CV (first submission)
- [ ] Ablation: do audio/prosodic features help?
- [ ] Fine-tune a transformer (DeBERTa) on transcripts
- [ ] Blend, final notebook report, plots

## Tech stack

Python, faster-whisper, language-tool-python, Hugging Face Transformers (GPT-2, CoLA RoBERTa), LightGBM, scikit-learn, pandas, matplotlib.