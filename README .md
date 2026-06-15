# 🎓 AI Grading Assistant

A multi-agent, agentic AI system for automated student answer grading — built with **LangGraph**, **Groq API**, and **Streamlit**.

> Final Project — Venture Track (AI Agent Development)
> Principles of AI – 001, Spring 2026, Woosong University

---

## 🌐 Live Demo

**App:** https://assessmentgrade-fe8k5vkfc4gzjw4xujbxgr.streamlit.app/

---

## ✨ Features

- **Two grading modes**
  - 📝 **Single Answer** — paste one student response, optionally upload a marking scheme for context
  - 📄 **Full Paper** — upload a question paper + a student answer sheet; the system extracts, matches, and grades every question automatically

- **6-agent grading pipeline** (LangGraph)
  - **Grader** — scores the answer 0–100 with a rationale
  - **Evidence** — extracts supporting points and missing information
  - **Integrity** — detects AI-generated content and plagiarism risk
  - **Bias Auditor** — checks the grade and evidence for fairness
  - **Review** — rule-based decision on whether human review is needed
  - **Explainer** — generates student-friendly feedback

- **Academic integrity detection**
  - AI-probability score, originality score, plagiarism risk, and a recommendation (Accept / Review / Reject)
  - Directly addresses AI-content detection requirements common in academic grading policies

- **Data management**
  - Every assessment is logged to `grading_log.csv`
  - Full paper reports can be exported as JSON
  - Assessment history viewable and downloadable in-app

- **File support**
  - Upload question papers / answer sheets as **PDF, DOCX, or TXT**

---

## 🏗️ Architecture

```
                ┌─────────────────────────────────────────┐
                │              Streamlit UI                │
                │   Single Answer  |  Full Paper modes     │
                └───────────────────┬───────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │            grading_engine.py             │
                │         (LangGraph State Graph)          │
                │                                           │
                │  Grader → Evidence → Integrity →          │
                │  Bias Auditor → Review → Explainer        │
                │                                           │
                │  Full Paper mode adds:                    │
                │  Extractor (parses Q paper)               │
                │  Matcher   (maps answers to questions)    │
                └───────────────────┬───────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │              Groq API                    │
                │   LLaMA 3.3-70B  ·  Qwen3-32B             │
                └─────────────────────────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │      grading_log.csv / JSON reports      │
                └─────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
.
├── app.py                  # Streamlit frontend (UI)
├── grading_engine.py        # LangGraph pipeline + agent logic (backend)
├── requirements.txt          # Python dependencies
├── grading_log.csv           # Auto-generated assessment history (created on first run)
└── README.md
```

---

## ⚙️ Setup & Local Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-grading-assistant.git
cd ai-grading-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a Groq API key
Sign up free at [console.groq.com](https://console.groq.com) and generate an API key.

### 4. Run the app
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser, then paste your Groq API key into the sidebar.

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push this repo to GitHub (`app.py`, `grading_engine.py`, `requirements.txt` in the root)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app**
3. Connect your repo, set **Main file path** to `app.py`, and deploy
4. Add your Groq API key under **⋮ → Settings → Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Every `git push` automatically redeploys the app

---

## 🧪 Usage

### Single Answer mode
1. (Optional) Upload a marking scheme / question file
2. Paste the student's answer
3. Click **Run Assessment**
4. Review results across the Grade, Integrity, Evidence, and Explanation tabs

### Full Paper mode
1. Upload the **Question Paper** (with marking scheme)
2. Upload the **Student Answer Sheet**
3. Click **Grade Full Paper**
4. View per-question results, overall score, and integrity flags
5. Download the CSV log or JSON report

---

## 🤖 Models Used

| Agent | Model |
|---|---|
| Grader | `llama-3.3-70b-versatile` |
| Evidence | `qwen/qwen3-32b` |
| Integrity | `llama-3.3-70b-versatile` |
| Bias Auditor | `qwen/qwen3-32b` |
| Review | Rule-based (no LLM) |
| Explainer | `llama-3.3-70b-versatile` |
| Extractor / Matcher (Full Paper mode) | `llama-3.3-70b-versatile` |

---

## ⚠️ Limitations & Future Work

- CSV-based logging is single-user; a database (e.g. Supabase/Postgres) would support concurrent access
- Integrity detection is heuristic (LLM-based), not a substitute for dedicated plagiarism-detection tools like Turnitin
- No authentication / multi-teacher accounts yet
- Batch grading for an entire class in one upload is a planned extension

---

## 📜 License

This project was developed for academic purposes as part of the Principles of AI course (Spring 2026), Woosong University.
