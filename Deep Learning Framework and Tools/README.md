# 🚀 Efficient BERT Fine-Tuning with LoRA on SST-2

This project demonstrates how to fine-tune a pre-trained BERT model using **parameter-efficient fine-tuning (PEFT)** via **LoRA (Low-Rank Adaptation)** on the SST-2 sentiment classification task, implemented entirely in a **Kaggle Notebook**.

We use 🤗 Hugging Face `transformers`, `datasets`, and `PEFT` libraries with PyTorch backend for this implementation.

---

## 🧠 Motivation

Fine-tuning large language models can be resource-intensive. LoRA enables efficient training by only updating a small number of additional parameters, which:
- Reduces training time
- Lowers memory usage
- Keeps the base model frozen

---

## 📓 Notebook Location

You can view and run the full notebook on Kaggle:

👉 **[Kaggle Notebook Link](https://www.kaggle.com/code/sudhinkarki/tangible2)**  


---

## 🧪 Dataset

We use the **GLUE SST-2** dataset:
> A binary sentiment classification dataset of movie reviews.  
Loaded using: `datasets.load_dataset("glue", "sst2")`

---

## 🧬 Model & Finetuning

- **Base model:** `bert-base-uncased`
- **Finetuning method:** LoRA via the `peft` library
- **Training framework:** Hugging Face `Trainer`

### LoRA Configuration:
```python
LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
```

---

## 🧠 Training Setup

- Subset of 2,000 training and 500 validation examples used for quick experimentation
- Batch size: 8
- Epochs: 3
- FP16 enabled
- Best model saved based on validation accuracy

---

## 📊 Results

| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|---------------|-----------------|----------|
| 1     | 0.6689        | 0.6788          | 0.530    |
| 2     | 0.5914        | 0.5938          | 0.654    |
| 3     | 0.5376        | 0.5145          | 0.770    |

---

## 📦 Model Artifacts

The trained model and tokenizer are saved to the `output/best_model` directory inside the notebook environment.

These can be used for inference like this:

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="output/best_model", tokenizer="output/best_model")
classifier("This movie was absolutely wonderful!")
```

---

## ✍️ Author

**Your Name**  
Sudhin Karki

👉 **[Blog Link](https://techgigsudhin20.hashnode.dev/fine-tuning-bert-efficiently-with-lora-for-sentiment-analysis-sst-2)** 

---

## 📄 License

MIT License
