![Tests](https://github.com/momedhat552/rag-assistant/actions/workflows/test.yml/badge.svg)

# RAG Assistant

Ask questions about your own documents. The app finds the most relevant passages and has the Gemini API answer using only that text, so it says "I don't know" instead of guessing when the answer isn't in the documents.

   **Live demo:** https://rag-assistant-xboenbprsnnhcy8qhmuv3y.streamlit.app/

## How it works

This is a small Retrieval-Augmented Generation (RAG) pipeline, built from scratch in plain Python:

1. **Chunk:** each document in `docs/` is split into overlapping pieces of about 500 characters.
2. **Embed:** every chunk is turned into an embedding (a list of numbers representing its meaning) with the Gemini embedding model, and saved to `index.json`.
3. **Retrieve:** the question is embedded the same way and compared with every chunk using cosine similarity. The top 3 chunks are kept.
4. **Generate:** the chunks and the question are sent to a Gemini model with the instruction to answer only from that context.

Because retrieval matches meaning instead of exact words, a question like "why does my code work only on my laptop?" finds the Docker document even though they share no keywords.

## Features

- Meaning-based search over `.txt` documents, with no vector database needed
- Answers grounded in the documents, with a refusal when the answer isn't there
- Command-line interface and a Streamlit web app
- Unit tests (pytest) and CI with GitHub Actions

## Tech stack

Python, Google Gemini API (embeddings and generation), Streamlit, pytest, GitHub Actions

## Project structure

```
rag-assistant/
├── rag.py              # chunking, embeddings, similarity search, answer generation, CLI
├── app.py              # Streamlit web app
├── test_rag.py         # unit tests
├── docs/               # the documents you can ask questions about
├── requirements.txt
└── .github/workflows/  # CI pipeline (runs pytest on every push)
```

## Setup

1. Clone the repo and go into the folder:

```
git clone https://github.com/momedhat552/rag-assistant.git
cd rag-assistant
```

2. Create and activate a virtual environment, then install the dependencies:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Mac/Linux, activate with `source .venv/bin/activate`.

3. Create a `.env` file with your Gemini API key:

```
GEMINI_API_KEY=your-key-here
```

## Usage

Build the index (run this once, and again whenever you change the files in `docs/`):

```
py rag.py index
```

Ask a question and get an answer:

```
py rag.py ask "why does my code work only on my laptop?"
```

Or just see the retrieved chunks with their similarity scores:

```
py rag.py "why does my code work only on my laptop?"
```

Run the web app:

```
streamlit run app.py
```

To use your own documents, put `.txt` files in `docs/` and rebuild the index.

## Tests

```
pytest
```

The tests cover chunking, cosine similarity, and prompt building. They don't call the API, so they run quickly and need no key. They also run automatically on every push through GitHub Actions.

## Deployment

The web app is deployed on Streamlit Community Cloud. The API key is stored in the app's Secrets settings and is never committed to the repository. `index.json` is not committed either, so the app builds it the first time it starts.

## Troubleshooting

- **`KeyError: 'GEMINI_API_KEY'`:** the API key isn't set. Locally, check that `.env` is in the project folder. On Streamlit Cloud, add `GEMINI_API_KEY = "your-key-here"` under the app's Secrets (a single line, with no `[section]` header above it) and reboot the app.
- **503 UNAVAILABLE:** the free Gemini tier is temporarily overloaded. Wait a minute and retry.
- **Answers seem out of date after editing `docs/`:** run `py rag.py index` again to rebuild `index.json`.

## Limitations and next steps

- Chunks are cut at a fixed length, so they sometimes break in the middle of a word or sentence. Splitting at sentence boundaries would improve retrieval.
- The "Retrieved from" line lists the files that were retrieved, not necessarily the ones the model used.
- Similarity search compares the question with every chunk, which is fine for a small set of documents but would need a vector database (such as Chroma) at larger scale.
- Only `.txt` files are supported. PDF upload is a natural next step.