# bot-index

A **local-first, CPU-friendly** semantic search and regression-test recommendation
system for **Automation Anywhere A360** bot libraries.

No cloud APIs. No LangChain. Runs entirely on a standard laptop.

---

## What it does

| Feature | Detail |
|---|---|
| **Parse** | Recursively traverses A360 bot JSON (commands, children, properties) |
| **Chunk** | Creates bot-summary, block-level, and action-level chunks with rich metadata |
| **Index** | Stores embeddings + metadata in a local [ChromaDB](https://www.trychroma.com/) persistent collection |
| **Search** | Cosine-similarity semantic search over all indexed chunks |
| **Recommend** | Combines vector search with a [NetworkX](https://networkx.org/) call graph for full impact analysis |

---

## Project structure

```
bot-index/
├── bots/                      ← Place your exported .json bot files here
├── chroma_db/                 ← Persistent ChromaDB storage (auto-created, gitignored)
├── example_bots/              ← 4 realistic interdependent demo bots
│   ├── Main_InvoiceProcessing.json
│   ├── LoginToCustomerPortal.json
│   ├── ProcessCustomerData.json
│   └── GenerateReport.json
├── src/
│   ├── __init__.py
│   ├── parser.py              ← A360 JSON parser + structure-aware chunker
│   ├── indexer.py             ← ChromaDB indexing logic
│   ├── graph.py               ← Build & query the bot call graph
│   ├── search.py              ← Semantic search + regression recommendation
│   └── utils.py               ← Shared helpers (embedding loader, ID gen, …)
├── main.py                    ← CLI entry point
├── config.py                  ← All knobs in one place
└── requirements.txt
```

---

## Quick-start

### 1 – Install dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

> **GPU acceleration** (optional): replace `torch` with the CUDA-enabled
> wheel from <https://pytorch.org/get-started/locally/> and set
> `EMBEDDING_DEVICE = "cuda"` in `config.py`.

### 2 – Index the example bots

```bash
python main.py index --bots-dir example_bots
```

Expected output:

```
Indexing bots from: …/example_bots
Indexing summary:
  GenerateReport: 13 chunk(s)
  LoginToCustomerPortal: 12 chunk(s)
  Main_InvoiceProcessing: 10 chunk(s)
  ProcessCustomerData: 15 chunk(s)

  Total: 4 bot(s), 50 chunk(s)
```

### 3 – Search

```bash
# Natural-language query
python main.py search "login button changed"

# With explicit UI element selectors
python main.py search "customer portal login page updated" \
  --ui-elements "button#login-submit,input#username"

# Disable call-graph expansion (direct matches only)
python main.py search "report export button" --no-graph
```

### 4 – Show the call graph

```bash
python main.py graph
```

### 5 – Check collection status

```bash
python main.py status
```

---

## Indexing your own bots

1. Export your A360 bots as JSON files (`.json`) from Automation Anywhere Control Room.
2. Drop them (or sub-folders of them) into the `bots/` directory.
3. Run:

```bash
python main.py index
```

Add `--force` to clear the existing index and rebuild from scratch:

```bash
python main.py index --force
```

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `BOTS_DIR` | `./bots` | Where to look for bot JSON files |
| `CHROMA_DB_DIR` | `./chroma_db` | ChromaDB persistent storage |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `EMBEDDING_DEVICE` | `cpu` | `cpu`, `cuda`, or `mps` |
| `CHROMA_COLLECTION` | `a360_bots` | Collection name inside ChromaDB |
| `DEFAULT_TOP_K` | `10` | Vector results per query |
| `MAX_GRAPH_HOPS` | `5` | Max transitive caller hops |
| `MAX_DISTANCE_THRESHOLD` | `0.6` | Max cosine distance (filter threshold) |
| `MAX_PARSE_DEPTH` | `20` | Max nested command depth |

### Swapping the embedding model

Edit `config.py`:

```python
# Higher quality, same size
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Code-aware model
EMBEDDING_MODEL = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
```

No other changes required – the model is loaded once and shared across all modules.

---

## How the chunking works

Each bot file produces three tiers of chunks:

```
Bot-level summary
└── Block chunks  (If/Else, Loop, …)
    └── Action chunks (Click, Set text, Run Task, …)
```

**Bot summary** – one per file:
```
Bot: LoginToCustomerPortal | Description: Handles secure login … | Variables: username, password, targetURL, loginStatus | Calls: bots/LoginToCustomerPortal | Windows: Customer Portal - Login, … | Selectors: input#username, input#password, button#login-submit, …
```

**Block chunk** – one per conditional/loop:
```
Bot: Main_InvoiceProcessing | Block: If (ConditionNode) | Contains: String: Assign [String], Run Task [TaskBot], …
```

**Action chunk** – one per leaf command:
```
Bot: LoginToCustomerPortal | Action: Click | Package: Recorder | Selector: button#login-submit | Selector Type: CSS | Window: Customer Portal - Login | Object Repository: CustomerPortal_LoginButton
```

---

## How regression recommendations work

1. **Vector search**: embed the query and find the most similar chunks.
2. **UI element matching**: if `--ui-elements` are supplied, run additional searches for each selector / window title.
3. **Evidence collection**: gather matching selectors, window titles, and action types per bot.
4. **Call-graph expansion**: for every directly matched bot, traverse the graph backwards to find all transitive callers.
5. **Scoring and ranking**: direct matches rank first (by accumulated cosine similarity); transitive callers follow (by inverse hop distance).

---

## Example bots

The four demo bots form an interdependency chain:

```
Main_InvoiceProcessing
├── calls → LoginToCustomerPortal
├── calls → ProcessCustomerData
└── calls → GenerateReport
```

This means that if `LoginToCustomerPortal` is affected by a UI change,
the tool will recommend both `LoginToCustomerPortal` **and**
`Main_InvoiceProcessing` for regression testing.

---

## A360 JSON format expectations

The parser handles the following A360 JSON keys:

| Key | Description |
|---|---|
| `commands` / `actions` | Array of command nodes (top-level) |
| `children` | Nested commands inside conditions/loops |
| `type` | Node type: `ActionNode`, `TaskBot: Run`, `ConditionNode`, `LoopNode`, … |
| `properties.commandName` | Human-readable action name |
| `properties.packageName` | A360 package (Recorder, Browser, …) |
| `properties.selector` | CSS / XPath selector |
| `properties.selectorType` | `CSS` or `XPATH` |
| `properties.windowTitle` | Target window title |
| `properties.objectRepositoryName` | Object Repository reference |
| `properties.taskBotPath` | Called bot path (for `TaskBot: Run`) |

---

## Troubleshooting

**"No .json bot files found"** → check `BOTS_DIR` in `config.py` or use `--bots-dir`.

**"Collection is empty"** → run `python main.py index --bots-dir example_bots` first.

**"Index schema version mismatch"** → your local index was built with an older
ID/schema strategy. Rebuild it with:

```bash
python main.py index --force
```

**Slow first run** → the embedding model (~90 MB) is downloaded on first use from HuggingFace Hub. Subsequent runs use the local cache (`~/.cache/huggingface`).

**Out of memory on large collections** → reduce `DEFAULT_TOP_K` in `config.py` or switch to a smaller model.

---

## Stack

| Library | Purpose |
|---|---|
| [sentence-transformers](https://www.sbert.net/) | Local text embeddings |
| [ChromaDB](https://www.trychroma.com/) | Local vector database (cosine similarity) |
| [NetworkX](https://networkx.org/) | Bot call graph |
| [tqdm](https://tqdm.github.io/) | Progress bars |
| Python stdlib | `pathlib`, `json`, `hashlib`, `argparse`, `logging` |
