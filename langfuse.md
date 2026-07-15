# Langfuse Integration — llm-multiroute

## What Was Changed

Three files were modified to add Langfuse observability:

| File | Change |
|------|--------|
| `llm-multiroute/requirements.txt` | Added `langfuse>=2.0.0` |
| `docker-compose.yml` | Added `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` env vars |
| `llm-multiroute/app/service/ai_service.py` | Switched to `get_client()` + `@observe` decorator on `_chat()` |

No other files were touched. All existing `metrics_store` calls remain in place alongside Langfuse.

> **SDK version note:** This integration uses the Langfuse v3/v4 API (`get_client()` + `@observe`).
> The v2 low-level API (`Langfuse().trace()` / `.generation().end()`) was removed in v3.
> If you see `AttributeError: 'Langfuse' object has no attribute 'trace'`, you are on v3+ — the code here is correct.

---

## How It Works

Every call to any endpoint (`/api/ai/classify`, `/api/ai/sentiment`, `/api/ai/summarize`, `/api/ai/intent`) flows through `_chat()` in `ai_service.py`. The instrumentation there:

1. `@observe(as_type="generation")` on `_chat()` — the decorator automatically opens and closes a **generation** span around the function, including timing
2. `langfuse.update_current_generation(...)` before the HTTP call — attaches the model name, prompt input, temperature
3. Ollama HTTP call runs (unchanged)
4. `langfuse.update_current_generation(...)` after the call — attaches the response text and token counts
5. Langfuse flushes the data to its backend asynchronously — zero added latency to your API

```python
from langfuse import get_client, observe

langfuse = get_client()   # reads LANGFUSE_* env vars automatically

@observe(as_type="generation")
def _chat(self, prompt, model, task_type="unknown"):
    langfuse.update_current_generation(
        name="ollama-chat",
        model=model,
        input=[{"role": "user", "content": prompt}],
        metadata={"temperature": self.temperature, "task_type": task_type},
    )

    # ... HTTP call to Ollama (unchanged) ...

    langfuse.update_current_generation(
        output=content,
        usage_details={"input": input_tokens, "output": output_tokens},
    )

    return content
```

---

## Setup

### Option A — Langfuse Cloud (recommended to start)

1. Sign up at [https://cloud.langfuse.com](https://cloud.langfuse.com)
2. Create a project → go to **Settings → API Keys**
3. Copy the **Public Key** (`pk-lf-...`) and **Secret Key** (`sk-lf-...`)
4. Add them to your `.env` file in the project root:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Option B — Self-hosted Langfuse (Docker)

Run Langfuse in a separate stack:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

Then in your `.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxx
LANGFUSE_HOST=http://localhost:3000
```

---

## Running the App

### Docker (production)

```bash
# Ensure .env in the project root contains all three LANGFUSE_* vars
docker-compose up --build
```

### Local dev

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com

cd llm-multiroute
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

---

## What You Can See in Langfuse

Once requests are flowing, open the Langfuse dashboard:

### Traces tab
- One trace per API request, automatically named after the function (`_chat`)
- Full timeline showing how long the Ollama generation took
- The exact prompt sent and response received — click any trace to inspect

### Generations tab
- Every LLM call across all tasks in one view
- Filter by model name (`gemma3:4b`, `ministral-3:3b`, etc.) to compare performance

### Metrics captured automatically
| Metric             | Where in Langfuse                                 |
|--------------------|---------------------------------------------------|
| End-to-end latency | Trace → duration                                  |
| Input token count  | Generation → Usage → Input                        |
| Output token count | Generation → Usage → Output                       |
| Estimated cost     | Generation → Cost (after model prices configured) |
| Model used         | Generation → Model column                         |
| Task type          | Generation → Metadata → task_type                 |
| Temperature        | Generation → Metadata → temperature               |
| Prompt content     | Generation → Input                                |
| Response content   | Generation → Output                               |

---

## Configuring Model Prices (for cost tracking)

Langfuse does not know Ollama model pricing by default. To see `$/request`:

1. In Langfuse → **Settings → Models**
2. Click **+ Add model**
3. For each model, fill in the name exactly as it appears in your `OLLAMA_MODEL_*` env vars:

| Model name       | Input price per 1M tokens | Output price per 1M tokens |
|------------------|---------------------------|----------------------------|
| `gemma3:4b`      | your cost                 | your cost                  |
| `ministral-3:3b` | your cost                 | your cost                  |
| `ministral-3:8b` | your cost                 | your cost                  |
| `gemma3:12b`     | your cost                 | your cost                  |

---

## Adding User Tracking (optional next step)

Currently traces are anonymous. To track per-user usage, add `user_id` to the current trace inside `_chat()`:

```python
@observe(as_type="generation")
def _chat(self, prompt: str, model: str, task_type: str = "unknown", user_id: str | None = None) -> str:
    langfuse.update_current_trace(user_id=user_id)
    langfuse.update_current_generation(...)
    ...
```

This requires adding `user_id: str | None = None` to the `TextRequest` DTO and threading it through the controller → service call. Once set, Langfuse's **Users** tab shows per-user token consumption and request history.

---

## Adding Prompt Management (optional next step)

Langfuse has a built-in prompt registry. Once prompts are stored there, you can edit them in the UI without redeploying and see which prompt version ran on each trace.

```python
# In classify_text(), replace the hardcoded prompt string with:
prompt_obj = langfuse.get_prompt("classify")          # name matches what you create in the UI
compiled_prompt = prompt_obj.compile(text=text)        # fills {{text}} placeholder

# Pass prompt_obj to update_current_generation so Langfuse links the version:
langfuse.update_current_generation(
    input=[{"role": "user", "content": compiled_prompt}],
    prompt=prompt_obj,
)
```

---

## Verifying the Integration

1. Send one request to any endpoint:
   ```bash
   curl -X POST http://localhost:8080/api/ai/classify \
     -H "Content-Type: application/json" \
     -d '{"text": "The stock market rose sharply today."}'
   ```

2. Open Langfuse → **Traces** tab
3. You should see a new trace with a generation named `ollama-chat` nested inside it

If no trace appears within 30 seconds, check:
- `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set and correct
- `LANGFUSE_HOST` matches your deployment (cloud vs self-hosted URL)
- App startup logs — `get_client()` will print a warning if keys are missing or authentication fails
