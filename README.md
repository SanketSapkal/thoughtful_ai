# Thoughtful AI – Support Agent

A lightweight, fully local **AI chat agent** built with **Streamlit** to simulate a customer support assistant for **Thoughtful AI**. The bot uses a simple **TF‑IDF + cosine similarity** algorithm to find the most relevant predefined response from a knowledge base of common questions.

No external APIs, libraries, or internet connectivity are required.

---

## 🧠 Features

- **Conversational Chat UI:** Streamlit-based chat interface (`st.chat_input`, `st.chat_message`).
- **Hardcoded Q&A Knowledge Base:** Includes answers about Thoughtful AI’s agents (EVA, CAM, PHIL).
- **Simple TF‑IDF Retrieval:** Retrieves best-matching answers using cosine similarity.
- **Offline Fallback:** If no strong match is found, it responds with a polite generic message.
- **Transparency:** Displays the matched question, source field, and similarity score.
- **Error Handling:** Gracefully handles empty or invalid input without crashing.

---

## 🚀 Quick Start

### 1️⃣ Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\\Scripts\\activate
pip install streamlit
```

### 2️⃣ Run the App
```bash
streamlit run app.py
```

Visit **http://localhost:8501** in your browser to start chatting with the agent.

---

## 💬 Example Questions
Try any of these to test the knowledge base:

- What does the eligibility verification agent (EVA) do?
- What does the claims processing agent (CAM) do?
- How does the payment posting agent (PHIL) work?
- Tell me about Thoughtful AI’s agents.
- What are the benefits of using Thoughtful AI’s agents?

If you ask an unrelated question, the agent will respond with a **generic fallback message**.

---

## 🧩 Architecture Overview

| Component | Description |
|------------|-------------|
| **KB (Knowledge Base)** | Hardcoded JSON-style Q&A pairs about Thoughtful AI |
| **TF‑IDF Engine** | Tokenizes, weights, and computes cosine similarity for retrieval |
| **Chat UI** | Built entirely with Streamlit’s chat components |
| **Fallback Handler** | Provides a default offline response when no match exceeds threshold |

---

## ⚙️ Configuration

- **Threshold:** The similarity cutoff (`SIM_THRESHOLD = 0.22`) controls how strictly the app decides a match. Increase this to make it pick fewer, higher-confidence answers.
- **No Internet Needed:** The app is fully self-contained and runs offline.

---

## 📂 Project Structure
```
app.py                 # Main Streamlit app
README.md              # Documentation (this file)
```

---

## 🧾 License
MIT License. You may use, modify, and distribute this project freely.

---

## 👨‍💻 Author
Created as part of a **Technical Screen** for building a simple AI support agent for **Thoughtful AI**.
