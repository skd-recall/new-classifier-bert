# news_classifier
# AG News Text Classification with Transformers

This project fine-tunes a pre-trained Transformer model (like BERT or DistilBERT) to classify news articles from the AG News dataset into one of four categories.

## 🔺 Dataset

The [AG News](https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) dataset contains news articles categorized into 4 classes:

* World
* Sports
* Business
* Sci/Tech

### Dataset Structure

Each row in the CSV files (`train.csv`, `test.csv`) follows this format:

```
class_index, title, description
```

**Note**: Class labels range from `1` to `4`. You may need to subtract 1 to match zero-based indexing (`0-3`).

## 📦 Requirements

Install dependencies:

```bash
pip install transformers datasets scikit-learn pandas torch tqdm
```

(Optional: For training logs)

```bash
pip install wandb
```

## 🧠 Training

You can train using either BERT or DistilBERT. Sample code snippet:

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
```

Adjust training arguments to fit your needs (epochs, batch size, etc.).

## 🚀 To Run

1. Download the dataset:

   * [AG News CSV (train/test)](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv)
2. Place `train.csv` and `test.csv` in the root folder.
3. Run your training script (`train.py` or Jupyter Notebook).

## 🧪 Evaluation

After training, the model is evaluated on the test set using accuracy, precision, recall, and F1-score.

## 📁 Folder Structure

```
.
├── train.csv
├── test.csv
├── train.py / notebook.ipynb
├── README.md
└── results/           # model checkpoints and logs
```

## 🏷️ Notes

* If using Hugging Face `datasets`, you can skip downloading CSVs and load directly with `load_dataset("ag_news")`.
* For faster training, consider using **DistilBERT** instead of BERT.

---

## 📬 Contact

For any questions or feedback, feel free to reach out!


