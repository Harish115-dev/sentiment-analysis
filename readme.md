# IMDB Sentiment Classification with BERT

Fine-tunes `bert-base-uncased` on the Stanford IMDB movie review dataset for binary sentiment classification (Positive / Negative) using the Hugging Face `transformers` and `datasets` libraries.

## Overview

This project:
- Loads the IMDB dataset (`stanfordnlp/imdb`) via Hugging Face `datasets`
- Tokenizes reviews using the BERT tokenizer
- Fine-tunes a `BertForSequenceClassification` model for 2 epochs
- Evaluates the model using accuracy and F1 score
- Provides a helper function to predict sentiment on new, arbitrary review text
- Saves the fine-tuned model and tokenizer to disk

## Requirements

```
numpy
torch
transformers
datasets
scikit-learn
```

Install with:
```bash
pip install numpy torch transformers datasets scikit-learn
```

A CUDA-capable GPU is recommended (the script automatically uses `fp16` and GPU acceleration when available, falling back to CPU otherwise).

## Dataset

- **Source:** `stanfordnlp/imdb` (loaded via `load_dataset`)
- **Train set:** full training split (25,000 labeled reviews)
- **Test set:** 500 examples, randomly sampled (shuffled with `seed=42`) from the original test split
- **Validation set:** a separate 200 examples (indices 500–700), shuffled with the same seed, drawn from the test split

Labels are binary: `0 = Negative`, `1 = Positive`.

## Configuration

| Parameter | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Max sequence length | 256 |
| Epochs | 2 |
| Learning rate | 2e-5 |
| Train/eval batch size | 8 |
| Weight decay | 0.01 |
| Seed | 42 |
| Eval strategy | per epoch |
| Logging steps | 50 |
| Mixed precision (fp16) | enabled if CUDA available |

## Pipeline Steps

1. **Setup & seeding** — sets random seeds across `random`, `numpy`, and `torch` (including CUDA) for reproducibility.
2. **Load dataset** — pulls the IMDB dataset and inspects a few sample texts/labels.
3. **Split data** — builds train, validation, and test subsets as described above.
4. **Tokenization** — applies the BERT tokenizer with truncation and a max length of 256 tokens to all three splits.
5. **Model setup** — loads `bert-base-uncased` with a classification head for 2 labels.
6. **Metrics** — computes accuracy and F1 score during evaluation.
7. **Training** — fine-tunes via Hugging Face `Trainer`, evaluating once per epoch on the validation set.
8. **Final evaluation** — runs `trainer.evaluate()` on the held-out 500-example test set and prints the results.
9. **Inference** — `predict_sentiment(review_text)` tokenizes a new review, runs it through the model, and returns the predicted label (`"Positive"`/`"Negative"`) plus class probabilities.
10. **Save model** — saves the fine-tuned model and tokenizer to `./model_bert`.

## Usage

Run the script/notebook cells in order. Key entry points:

**Train the model:**
```python
trainer.train()
```

**Evaluate on the test set:**
```python
final_result = trainer.evaluate(eval_dataset=test_tz)
print(final_result)
```

**Predict sentiment on a new review:**
```python
predict_sentiment("this movie i dont know but i dont really like it although is good but not my type")
# Returns: (sentiment_label, probabilities_array)
```

**Save the trained model:**
```python
trainer.save_model("./model_bert")
tokenizer.save_pretrained("./model_bert")
```

## Output

- Training checkpoints and logs are written to `./imdb output`
- The final fine-tuned model and tokenizer are saved to `./model_bert`

## Notes

- Training and evaluation metrics (loss, accuracy, F1) are printed during training (every 50 steps) and at the end of each epoch.
- The final test-set evaluation results (loss, accuracy, F1) are printed after training completes.