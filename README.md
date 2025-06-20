# RAG-QnA Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for answering questions about the CNN/DailyMail news articles using hybrid search and Google Gemini LLM.

---

## 🚀 Features

- Hybrid search: combines dense, sparse, and late interaction embeddings for robust retrieval
- RAG pipeline: retrieved documents are used as context for LLM answers
- Streamlit frontend: interactive chat interface with source document display
- Adjustable retrieval settings and search method

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

- Copy `.env.example` to `.env`:
  ```bash
  cp .env.example .env
  ```
- Fill in your actual API keys for `GOOGLE_API_KEY` and `QDRANT_API_KEY` in the `.env` file.

### 4. Run the Streamlit App

```bash
streamlit run chatbot_app.py
```

---

## ⚙️ Configuration Details

- **API Keys:**
  - `GOOGLE_API_KEY`: Your Google Gemini API key
  - `QDRANT_API_KEY`: Your Qdrant Cloud API key
- **Qdrant Collection:**
  - Name: `hybrid-search4`
  - Stores dense, sparse, and late interaction vectors for each article
- **Models Used:**
  - Dense: `all-MiniLM-L6-v2`
  - Sparse: `BM25`
  - Late Interaction: `ColBERT v2.0`
  - LLM: `Google Gemini 2.0 Flash`

---

## 💬 Sample Queries

Try asking questions like:

- Who is the current president of Nicaragua?

  - Answer: The current president of Nicaragua is Daniel Ortega.
- How much money did Daniel Radcliffe gain access to?

  - Answer: Daniel Radcliffe gained access to a reported £20 million ($41.1 million) fortune.
- What happened between Nicaragua and Colombia?

  - Answer: Nicaragua broke diplomatic relations with Colombia "in solidarity with the Ecuadoran people," after Colombia attacked a rebel camp in Ecuador.
- Give me a summary of the main events in the CNN/DailyMail articles.

  - Answer: Here is the summary of the main events covered in the CNN articles:

  * **I-Report Anniversary:** CNN's I-Report initiative, launched in August 2006, marked its first anniversary. It has become an integral part of CNN's news coverage, with citizens contributing photos, videos, and information during major breaking news events.
  * **Minneapolis Bridge Collapse:** I-Reporters responded to the collapse of a bridge over the Mississippi River in Minneapolis, with Mark Lacroix providing photos and live updates.
  * **Virginia Tech Shooting:** Jamal Albarghouti sent cell phone video of the Virginia Tech shooting.
  * **Dallas Gas Facility Explosion:** I-Reporters recorded video of explosions at an industrial gas facility in Dallas, Texas.
  * **New York Steam Pipe Explosion:** Jonathan Thompson sent video of a steam pipe explosion in New York and later footage of the repairs.
  * **Iraqi Women's Hardships:** CNN correspondent Arwa Damon reports on the hardships faced by Iraqi women, including stories of loss, torture, and resilience in the face of violence. The article highlights the stories of Nahla, whose husband was killed in a bomb blast, Dr. Eaman, a children's doctor who has been attacked by insurgents, and Yanar, who started an organization for women's freedom in Iraq.
  * **CNN Boardroom Master Classes:** CNN is hosting three events in Shanghai, New York, and London to share modern business practices. Howard Schultz, chairman of Starbucks, will be a guest at the Shanghai event.

---

## 📄 Project Structure

```
├── chatbot_app.py         # Streamlit frontend
├── requirements.txt       # Python dependencies
├── .env 		   # Environment variable template
├── README.md              # Project documentation
├── report.md              # Technical report
├── data/
│   └── cnn_dailymail_train_1000_articles.csv
└── test.ipynb            # Development notebook
```

---

## 📝 How it Works

1. User enters a question in the chat interface.
2. The system retrieves relevant articles using hybrid search (dense, sparse, late interaction embeddings).
3. Retrieved articles are passed as context to the Google Gemini LLM.
4. The LLM generates an answer grounded in the retrieved context.
5. Sources and scores are displayed alongside the answer.

---

## 🙌 Credits

- Built with [Streamlit](https://streamlit.io/), [Qdrant](https://qdrant.tech/), [Sentence Transformers](https://www.sbert.net/), [ColBERT](https://colbert-ai.github.io/), and [Google Gemini](https://ai.google.dev/gemini-api).

---

## 📢 License

Add your license information here.
