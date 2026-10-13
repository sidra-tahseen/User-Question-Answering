# Context QA: Where Does Fine-Tuned DistilBERT Get the Answer Wrong?

Built as a question-answering project for a data science portfolio.


**Live Demo:** [Try the Question Answering App](https://user-question-answering-gdhpbb2syrsi8u2v5cuzka.streamlit.app/) · **Model:** [Hugging Face — User-Questioning-Answer](https://huggingface.co/sidratahseen/User-Questioning-Answer)


I implemented a TF-IDF sentence-retrieval baseline, fine-tuned DistilBERT for extractive question answering on a subset of SQuAD 1.1, compared it with an existing SQuAD-fine-tuned checkpoint, and inspected incorrect predictions to understand where answer-span selection breaks.

**Live demo:** TODO (Streamlit URL) &nbsp;|&nbsp; **Model:** TODO (Hugging Face model URL)

## Key findings

1. **Fine-tuning substantially outperformed sentence retrieval.** On the same 1,000-example validation subset, the TF-IDF baseline recorded 0.40% Exact Match and 19.32% token F1, while the selected fine-tuned DistilBERT run recorded 61.80% EM and 72.75% F1.
2. **More training data helped in the recorded experiments.** The selected run used 20,000 training examples for two epochs, compared with an earlier 10,000-example run. The earlier model's reported validation scores varied across evaluation runs, so this comparison should be treated as indicative rather than a controlled estimate of the effect of data size alone.
3. **A lower learning rate performed worse in this setup.** Training from the base checkpoint on the same 20,000-example subset for two epochs at `1e-5` recorded 52.30% EM and 63.26% F1, below the selected run at `2e-5`.
4. **An extra epoch did not improve the selected model's scores.** Continuing the selected model for one more epoch at `1e-5` recorded 60.40% EM and 72.13% F1.
5. **Errors were often span-selection errors, not missing information.** Examples included predicting “lower” instead of “Last Glacial Maximum,” selecting “34 hours” instead of “over 3 days,” and returning one person from a passage when the question asked for another.
6. **Longer answers were harder to match exactly in the inspected evaluation.** Exact Match decreased from 62.72% for one-token answers to 40.32% for answers of six or more tokens. This is a descriptive result from the evaluated subset, not a universal claim about QA systems.

## Results

### Main comparison (1,000-example SQuAD validation subset)

| Approach | Exact Match (EM) | Token F1 |
|---|---:|---:|
| TF-IDF sentence retrieval | 0.40% | 19.32% |
| Fine-tuned DistilBERT — selected experiment | 61.80% | 72.75% |
| Existing SQuAD-fine-tuned DistilBERT reference checkpoint | 79.20% | 86.58% |

The reference checkpoint (`distilbert/distilbert-base-cased-distilled-squad`) had already been fine-tuned on SQuAD. It is included as a reference benchmark, not as an independently trained or untouched model. Because it was trained on the same dataset family used for evaluation, its score should not be interpreted as a clean estimate of performance on unseen datasets.

### Training experiments

All custom experiments used the same selected 20,000-example training subset for the runs listed below, unless noted otherwise. The base checkpoint was `distilbert/distilbert-base-cased`.

| Experiment | Training setup | Recorded EM | Recorded token F1 |
|---|---|---:|---:|
| Earlier custom model | 10,000 examples; initial run followed by further training | 57.90% | 70.14% |
| Experiment 2 — selected | 20,000 examples; 2 epochs; learning rate `2e-5` | 61.80% | 72.75% |
| Experiment 2 — repeat evaluation | Same saved model and 1,000-example evaluation subset | 62.00% | 72.81% |
| Experiment 3 | 20,000 examples; 2 epochs; learning rate `1e-5` | 52.30% | 63.26% |
| Experiment 4 | Continued Experiment 2 for one additional epoch at `1e-5` | 60.40% | 72.13% |

Repeated evaluation produced slightly different results for the selected model (61.80/72.75 versus 62.00/72.81). This discrepancy has not been fully explained, so the scores are reported as recorded rather than presented as perfectly reproducible measurements. The selected checkpoint was retained because it had the highest recorded EM and F1 among these experiments.

### Baseline details

The baseline splits each context into sentences, vectorizes the question and sentences with TF-IDF, and returns the sentence with the highest cosine similarity to the question. It returns a sentence rather than predicting an answer span, which helps explain its very low Exact Match score. Token F1 gives partial credit when the returned sentence contains some of the reference answer tokens.

### Error analysis

Inspection of incorrect predictions identified several recurring patterns:

| Question type / example | Reference answer | Prediction | Observed failure |
|---|---|---|---|
| Abbreviation definition: “What does LGM stand for?” | Last Glacial Maximum | lower | Missed a definition that appears in the passage |
| Duration: extended Lunar Module surface stay | 3 days | 34 hours | Chose a related value from a different sentence |
| Person identification: alumni who was Obama's campaign adviser | David Axelrod | Saul Alinsky | Selected a nearby entity instead of the requested person |
| Multi-part answer: project structures assisting integration | design-build, partnering and construction management | Several project structures | Returned a broad introductory phrase rather than the list |

In the examined 1,000-example subset, Exact Match by reference-answer length was:

| Reference answer length | Exact Match |
|---|---:|
| 1 token | 62.72% |
| 2–3 tokens | 60.09% |
| 4–5 tokens | 56.52% |
| 6+ tokens | 40.32% |

This breakdown uses the first reference answer's token length, while the scoring code takes the best EM/F1 over the available references. It is exploratory analysis on one subset; it does not establish that answer length alone causes the errors.

## Method

- **Dataset:** SQuAD 1.1 via Hugging Face Datasets (`rajpurkar/squad`).
- **Training subset:** 20,000 examples selected from the training split.
- **Evaluation subset:** 1,000 examples selected from a shuffled validation split.
- **Base model:** `distilbert/distilbert-base-cased`.
- **Selected training configuration:** 2 epochs, learning rate `2e-5`, batch size 8, weight decay `0.01`, maximum sequence length 384, document stride 128.
- **Preprocessing:** Questions and contexts are tokenized as pairs. Long contexts can overflow into overlapping windows. Training labels identify the answer start and end token positions; windows that do not contain the answer use the CLS position.
- **Baseline:** TF-IDF sentence retrieval with cosine similarity.
- **Metrics:** Normalized Exact Match and token-level F1, taking the maximum score across the reference answers for each question.
- **Inference:** Hugging Face Transformers question-answering pipeline.
- **Interface:** Streamlit.

## Limitations

- The model can return an incorrect or incomplete answer even when the correct information is present in the passage.
- The selected custom model was trained on only 20,000 examples from SQuAD 1.1, not the full training set.
- Repeated evaluation scores varied slightly, and the source of the variation has not been fully resolved.
- The reference checkpoint was already fine-tuned on SQuAD and is not an independent generalization benchmark.
- The validation subset comes from SQuAD, so these results do not establish performance on unrelated domains.
- Exact Match is strict: a prediction can be semantically close but still receive zero if it does not match a reference after normalization.
- The app's displayed pipeline score is not a calibrated probability that an answer is correct.
- The system extracts a span from the supplied context. It does not browse for facts and should not be treated as a guarantee of factual correctness.

## Run it yourself

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the model files

Place the saved model and tokenizer files in a directory named `qa-distilbert-best/` beside `app.py`. The directory should contain the model configuration, model weights, and tokenizer files.

The model weights are not included in this repository by default because they can be large. A hosted model link will be added here once the model is published.

### 3. Run the app

```bash
python -m streamlit run app.py
```

## Live Demo

Try the deployed application: **[Find Answers in Your Text](https://user-question-answering-gdhpbb2syrsi8u2v5cuzka.streamlit.app/)**

The app accepts a passage and a question, then uses a fine-tuned DistilBERT extractive question-answering model to identify the answer span within the supplied text. It does not search the web or generate information beyond the passage.

### How to use

1. Open the live demo.
2. Paste a passage into the context field.
3. Enter a question whose answer is contained in the passage.
4. Submit the question to view the predicted answer and model score.

### Model

The application loads the fine-tuned model hosted on Hugging Face:

[**View the model repository**](https://huggingface.co/sidratahseen/User-Questioning-Answer)

### Deployment

- **Frontend:** Streamlit
- **Model hosting:** Hugging Face Hub
- **Inference:** Hugging Face Transformers
- **Model architecture:** DistilBERT for extractive question answering


Streamlit will print a local URL, usually `http://localhost:8501`.

## Repository layout

```text
app.py                    Streamlit question-answering interface
requirements.txt          Python dependencies
qa-distilbert-best/       Saved model and tokenizer (local; not committed by default)
README.md                 Project overview, results, methodology and limitations
```

## Future work

- Resolve evaluation reproducibility and save a fixed set of predictions.
- Evaluate on an independent dataset or a different domain.
- Compare answer-span post-processing strategies.
- Investigate long-context and multi-part-answer failures.
- Add visualizations for evaluation metrics and error categories.
- Publish the model artifact and deploy the Streamlit demo.

## Acknowledgements

- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Streamlit](https://docs.streamlit.io/)
