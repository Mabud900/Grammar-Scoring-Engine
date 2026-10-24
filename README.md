Grammar Scoring Engine for Spoken English
A machine-learning regression pipeline for predicting the grammar score
of a spoken English audio clip on the SHL Hiring Assessment 2026
dataset.
Project overview
The goal is to map each .wav recording to a continuous grammar score
using the labels supplied in train.csv.

The central design is a feature-based regression pipeline. Whisper
and the grammar/language models are used as frozen feature extractors;
they are not fine-tuned for this project. The only model trained on the
labeled examples is LightGBM.
audio.wav
   |
   v
Whisper ASR (frozen)
   |--------------------> transcript + word timestamps
   |                                  |
   |                    Frozen text scorers / feature extractors
   |                    - LanguageTool grammar checks
   |                    - GPT-2 perplexity
   |                    - CoLA-trained BERT acceptability score
   |                    - transcript statistics
   |                    - optional timing / acoustic features
   |                                  |
   +----------------------------------+
                                      v
                         One numeric feature row per clip
                                      |
                                      v
                         LightGBM regression (trained)
                                      |
                                      v
                            Predicted grammar score
                                  (0 to 5)
For each recording, the feature-extraction stage converts the transcript
and, where implemented, audio timing information into a row of numeric
values. The training labels are joined to those rows by filename.
LightGBM learns the relationship between the features (X) and the
known scores (y).
Important: Whisper is an automatic speech-recognition system, not
the grammar-scoring model. It generates the transcript. The downstream
features and LightGBM model estimate the score.

Dataset
The competition description specifies:
- Training set: 769 labeled audio samples.
- Test set: 216 audio samples for which predictions must be
  generated.
- Audio format: WAV.
- Target: a grammar score on a 0--5 scale. The supplied rubric
  describes scores 1--5, but the training data has also been observed
  to contain 0.0 labels. Inspect the actual labels rather than
  assuming the rubric tells the whole story.
- Evaluation metrics: RMSE and Pearson correlation.
- Notebook requirement: report training RMSE. Out-of-fold (OOF)
  cross-validation RMSE should also be reported because it is a more
  useful estimate of generalization.
Typical data files:
project/
├── README.md
├── grammar_scoring_pipeline.ipynb
├── train.csv
├── test.csv
├── sample_submission.csv
├── train/
│   ├── audio_....wav
│   └── ...
└── test/
    ├── audio_....wav
    └── ...
The exact folder layout may differ. Update the notebook's data paths to
match your Drive or Kaggle layout.
CSV roles
- train.csv: audio filenames and their ground-truth grammar labels.
- test.csv: filenames for the clips that need predictions. The
  competition description warns that its label values are
  placeholders/random; do not use the test label column as training
  truth.
- sample_submission.csv: the required output format. Preserve its
  column names and row order when creating the final submission.
The CSV and WAV files are linked by filename. Verify that every filename
points to an existing audio file and check for missing or duplicate
entries before feature extraction.
Method
1. Audit and validate the data
Before training:
1. Load train.csv and test.csv.
2. Check shapes, column names, missing values, duplicate filenames, and
   label distribution.
3. Confirm that each referenced WAV file exists.
4. Inspect audio duration and sample rate.
5. Listen to representative recordings and compare selected transcripts
   with the audio.
A data audit found a group of 37 training files named in the
audio_50xx.wav range with label 0.0, while the remaining training
files were labeled between 1.0 and 5.0. This is an observation about
the supplied training data, not proof of what those recordings contain.
Inspect them before deciding how to handle them. The test filename check
described in the project notes found no 50xx files; verify again
against the current files.
2. Transcribe audio with Whisper
Run Whisper on the training and test recordings to obtain transcripts
and, where enabled, word-level timestamps. Save intermediate results to
disk so a disconnected Colab/Kaggle session does not force all clips to
be transcribed again.

Whisper is kept frozen. Its output is used as input to later feature
extraction; Whisper is not trained against the grammar labels.

Known limitation: ASR may omit disfluencies or normalize a speaker's
ungrammatical wording. Compare transcripts with the original audio,
especially for examples with low and high labels, to understand how much
grammar information survives transcription.
3. Extract numeric features
The planned text feature groups include:
- LanguageTool: grammar/spelling issue counts, normalized by
  transcript length where appropriate.
- GPT-2: text perplexity as a measure related to how
  predictable/fluent the word sequence is.
- CoLA-trained BERT: grammatical acceptability-related score.
- Transcript statistics: word count, sentence count, average
  sentence length, vocabulary variety, repetitions, and filler words.
- Timing/audio features, if enabled: words per minute, pause
  count/duration, pause ratio, and acoustic measurements.
Not every feature group is guaranteed to be implemented in every version
of the notebook. The notebook is the source of truth for the features
actually computed. Handle empty transcripts and failed model outputs
explicitly, and make sure every feature is numeric and finite before
fitting LightGBM.
4. Build the feature table
Create one row per recording, keyed by filename, with all extracted
feature columns. Join training labels from train.csv using that key.

Conceptually:
  filename             grammar text feature       timing          ...        label
                     feature 1            2    feature 3              
  audio_001.wav            ...          ...          ...          ...          3.5
  audio_002.wav            ...          ...          ...          ...          4.0
The table above is illustrative; values depend on the feature extraction
run.

Use the same feature-generation logic and column order for train and
test data. Do not include label as an input feature.
5. Train LightGBM regression
Let X be the numeric feature matrix and y the labels from
train.csv. LightGBM is the trainable model:
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
The frozen models supply pretrained knowledge; LightGBM learns how to
combine the resulting signals for this particular dataset. With only 769
labeled samples, use conservative model settings and cross-validation
rather than assuming a complex model will generalize.
6. Evaluate with cross-validation
Use five-fold cross-validation, keeping the label distribution
reasonably similar across folds (for example, by splitting on binned
labels when suitable). Generate OOF predictions so each training sample
is predicted by a model that did not train on that sample.

Report at least:
- Training RMSE: required by the competition instructions,
  calculated from a model fitted on the training data.
- OOF RMSE: a more realistic estimate of performance on unseen
  samples.
- Pearson correlation: the other competition metric.
- Baseline comparison: compare against a simple mean-label
  predictor.
- Optional diagnostics: predicted-versus-actual plot, residual
  plot, and feature importance.
Training RMSE is usually optimistic. A large gap between training RMSE
and OOF RMSE can indicate overfitting. If you add or remove a feature
group, compare OOF results to determine whether it actually helps.
7. Predict and create the submission
1. Fit the final LightGBM model using all labeled training rows and the
   selected features.
2. Generate the same features for each test recording.
3. Predict the test scores.
4. Clip predictions to the task's valid range, based on the
   competition/data requirements.
5. Write predictions in the exact sample_submission.csv structure,
   preserving its expected filenames, columns, and row order.
6. Check the output for missing values, duplicate/missing filenames,
   and an incorrect number of rows before submitting.
Do not use the placeholder labels in test.csv to train or tune the
model.
Running in Google Colab
1. Open grammar_scoring_pipeline.ipynb in Colab.
2. Mount Google Drive:
   from google.colab import drive
   drive.mount('/content/drive')
3. Set the data directory in the notebook to the folder containing the
   CSV files and audio directories. For example:
   DATA_DIR = "/content/drive/MyDrive/SHL_Hiring_Assessment_2026"
   This is only an example path; change it to match your actual Drive
   folders.
4. Install the dependencies listed or imported by the notebook. Typical
   dependencies for this approach may include pandas, numpy,
   scikit-learn, lightgbm, faster-whisper, transformers, and
   the libraries required by LanguageTool and any optional audio
   features. Use the notebook's actual imports and setup cells as the
   authoritative dependency list.
5. Run the data audit cells first. Then run transcription, feature
   extraction, cross-validation, final fitting, and submission
   generation in order.
6. Confirm that intermediate transcripts/features are saved and can be
   reused after a runtime restart.
A GPU can speed up Whisper and transformer-based feature extraction, but
the final LightGBM training stage is comparatively lightweight.
Reproducibility and data leakage safeguards
- Set random seeds for model training and cross-validation.
- Keep the test set out of model fitting and validation decisions.
- Fit any data-dependent preprocessing only on each training fold, not
  on the entire dataset before cross-validation.
- Join labels by filename, never by row position alone.
- Cache expensive transcript and feature outputs with a stable
  filename key.
- Keep a results table containing model/features, OOF RMSE, and
  Pearson correlation.
- Do not assume similarly named files in separate train/test folders
  are the same recording. If file identity matters, compare the actual
  files rather than filenames alone.
Limitations
- The training dataset is small, so performance estimates can vary by
  fold.
- Whisper transcription errors can hide grammar errors.
- Language-model scores are proxies for grammar, not perfect
  measurements of a human rating rubric.
- The label distribution is uneven, with very few low-scoring speech
  examples in the observed training data.
- Acoustic features may or may not improve performance; retain them
  only if validation supports the added complexity.
- Any observed filename/label pattern should be checked against the
  audio and the competition's data-generation process before it is
  treated as a general rule.
