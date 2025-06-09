# FactuAI

**FactuAI** is a full-stack AI-powered application for:
- 📰 **News summarization**
- 🔍 **Fact-checking political claims**

Built as part of my **Final Year Project** in Computer Science (AI) at APU, Malaysia.

---

## 🎯 Goals

- Extract concise summaries from long-form news articles
- Automatically classify factual accuracy of statements (e.g. `true`, `false`, `pants_on_fire`)
- Empower responsible media consumption using NLP and deep learning

---

## 🛠️ Tech Stack

| Layer       | Tech                                      |
|-------------|-------------------------------------------|
| Frontend    | React (Next.js), Tailwind CSS, Shadcn UI  |
| Backend     | Flask (REST API), SQLAlchemy              |
| Database    | PostgreSQL                                |
| ML Models   | Hugging Face Transformers (BERT, T5)      |
| Tools       | PyTorch, Optuna, xFormers, Jupyter        |

---

## 🧠 ML Models

- **Summarization**: fine-tuning T5 on [MultiNews](https://huggingface.co/datasets/alexfabbri/multi_news)
- **Fact-checking**: fine-tuning BERT and NeoBERT on [LIAR2](https://github.com/chengxuphd/liar2)

All models are trained with optimized preprocessing and hyperparameter tuning (Optuna).

---

## 📌 Current Status

- ✅ Dataset preprocessing (`LIAR2`, `MultiNews`)
- ✅ Tokenization, padding, vocabulary
- ✅ Model fine-tuning with Hugging Face
- ✅ React-based frontend with login/register
- ✅ Flask backend with PostgreSQL integration
- ⏳ In progress: dashboard summarization + fact-check endpoint
- ⏳ In progress: model evaluation, export, deployment

---

## 📁 Folder Structure

```bash
FactuAI/
├── backend/          # Flask app
├── frontend/         # Next.js app
├── data/             # Processed CSVs
├── models/           # Saved Hugging Face checkpoints
├── notebooks/        # Training notebooks (not tracked)
├── scripts/          # Training scripts
├── README.md
```

---

## 🛡️ License

Licensed under the [MIT License](LICENSE).  
Free for academic, research, and learning purposes.

---

## 👨‍💻 Author

**Zayed Ramadan Rahmat**  
Final Year BSc (Hons) Computer Science (AI), APU  
📍 Kuala Lumpur, Malaysia  
🔗 [LinkedIn](https://linkedin.com/in/zayedrmdn) · 📧 [Email](mailto:zayedrmdn@email.com)
