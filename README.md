# RAG Demo — Spring AI + Ollama

A minimal **Retrieval-Augmented Generation (RAG)** demo built with Spring Boot and Spring AI,
running a local LLM through [Ollama](https://ollama.com) (no API key, no cost).

It answers questions about a FAQ document by retrieving the relevant parts first,
then letting the model answer **grounded on that content** instead of hallucinating.

## How RAG works here

```
INGEST (once at startup)
  faq.txt --> split into chunks --> embed each chunk --> store in vector store

QUERY (per question)
  question --> embed --> similarity search (top-k chunks)
            --> inject chunks into the prompt as context
            --> LLM answers from that context
```

1. **Retrieve** — the question is embedded and compared by meaning against the stored
   chunks; the closest ones are selected (`topK`).
2. **Augment** — the retrieved chunks are inserted into the prompt as context.
3. **Generate** — the model answers using only that context.

The console prints each stage (`RETRIEVED`, `FINAL PROMPT`, `ANSWER`) so the flow is visible.

## Tech stack

| Part            | Choice                                             |
|-----------------|----------------------------------------------------|
| Framework       | Spring Boot 3, Spring AI 1.0                        |
| LLM             | Ollama `llama3.2` (local)                           |
| Embeddings      | Ollama `nomic-embed-text` (local)                   |
| Vector store    | `SimpleVectorStore` (in-memory)                     |
| Document reader | Apache Tika (`TikaDocumentReader`) — reads txt/docx/pdf |

## Prerequisites

1. Java 17+ and Maven
2. [Ollama](https://ollama.com) installed and running
3. Pull the models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

## Run

```bash
mvn spring-boot:run
```

The app runs the ingestion + one question, prints the stages, and exits
(it has no web server — `spring.main.web-application-type=none`).

Example output:

```
Loaded 7 chunks from faq.txt

--- RETRIEVED (what vector search found) ---
* How long do refunds take? Refunds are processed within 2 to 10 business days...

--- FINAL PROMPT (what the model actually sees) ---
Answer using ONLY the context below...

--- ANSWER ---
Refunds take 2 to 10 business days after the item reaches the seller.
```

## Try it

Edit the `question` in `RagDemoApplication` and re-run:

- `"What payment methods do you accept?"` → retrieval now returns the **payment** chunk.
- `"Do you deliver to the moon?"` → the model should answer **"I don't know"** (not in the FAQ) — grounding prevents hallucination.
- Change `TOP_K` to see how many chunks are retrieved.

## Project structure

```
src/main/java/com/demo/RagDemoApplication.java   # the whole RAG flow
src/main/resources/faq.txt                       # the knowledge source
src/main/resources/application.yml               # Ollama + model config
```

## Notes

- Ollama runs as a background service on `http://localhost:11434`; the app talks to it over HTTP.
- `SimpleVectorStore` is in-memory and resets on restart. For persistence, swap in a
  real vector store such as **pgvector** (PostgreSQL) or Elasticsearch.
- If you switch to a cloud model (OpenAI, Anthropic), keep the API key out of the repo
  (environment variable + `.gitignore`).
