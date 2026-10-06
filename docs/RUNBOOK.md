# Hackathon Runbook — 06 Oct 2026

Timings from the official schedule. The point of this page is that nobody has to
think about *process* while the clock is running.

| Time | Phase | What the squad does |
| --- | --- | --- |
| 09:00 | Attendance & squad formation | — |
| 09:30 | **Use case explained** | Listen. Write down the nouns. Do not open an editor. |
| 10:00 | Sprint 0 — design & planning | Adapt the diagram in `docs/ARCHITECTURE.md` |
| 11:00 | Sprint 0 — design presentation | 10–15 min per squad, get SME sign-off |
| 11:30 | Sprint 1 — implementation | Build against the signed-off design |
| 13:00 | Working lunch | Keep an eval running |
| 16:00 | **Freeze** | Stop building. Dry-run the demo twice. |
| 16:30 | Sprint review | 20 min per squad |

## 09:30 — while the use case is explained

Capture these five things. They determine everything downstream.

1. **What are the documents?** Format, language, scanned or digital, how many.
2. **Who asks the questions?** And what does a good answer look like to them?
3. **What is the single demo-able outcome?** The one thing that must work at 16:30.
4. **Is anything out of scope?** Write it down — it goes on the design slide.
5. **How is it judged?** Accuracy, speed, UI, novelty. Optimise for the rubric.

## 10:00 — Sprint 0, two hours

Do not write pipeline code. The pipeline exists. Spend the time on:

- [ ] Edit the flow diagram in `docs/ARCHITECTURE.md` to use the real nouns
- [ ] Fill in `## Problem` and `## Approach` in `README.md`
- [ ] Replace `eval/questions.json` with 8–10 questions from the **real** use case
- [ ] List what you are deliberately not building
- [ ] Decide who speaks for the 10–15 minutes

The eval set is the highest-value item on that list. Write the questions before
the code — they are the definition of done.

## 11:30 — Sprint 1, adapting the template

```bash
# 1. point it at the real documents
python -m rag ingest path/to/their/files

# 2. does retrieval find the right things? (free, seconds)
python -m rag eval --retrieval-only

# 3. only once retrieval is good, spend tokens on answers
python -m rag eval
```

**Fix retrieval before touching prompts.** An answer cannot be better than the
chunks it was given. If `hit@k` is low, no prompt engineering will save it.

Common adaptations, in likely order of need:

| If the use case is... | Change |
| --- | --- |
| A different file format | Add a loader in `rag/loaders.py`, follow the existing shape |
| Not English | `EMBED_PROFILE=multilingual`, `OCR_LANG=...` |
| Needs structured output | Add a JSON-schema instruction in `pipeline.SYSTEM_PROMPT` |
| Needs conversation memory | `app.py` already keeps history; feed it into the prompt |
| Needs aggregation/counting | Raise `TOP_K`, or compute in pandas and put the result in context |
| Needs sources shown to users | Already there — `answer.sources` and the Sources expander |

## 16:00 — freeze

Stop building. Then, in order:

- [ ] `python -m rag eval` — record the numbers, put them on a slide
- [ ] `python -m pytest tests/ -q` — should be green
- [ ] Run the demo end to end, **twice**, on the machine you will present from
- [ ] Rehearse the refusal case — "it says I don't know" is a feature, show it
- [ ] `git push`
- [ ] Phone hotspot on, laptop charged, charger in the bag

## When something breaks

| Symptom | Fix |
| --- | --- |
| `429` / rate limit | Already automatic: Groq 120b → Groq 20b → Gemini |
| Both keys dead | Open `notebooks/colab_rag.ipynb`, fresh quota |
| Laptop dies | Same notebook, any machine, repo clones from GitHub |
| Index won't load | `python -m rag ingest` — it rebuilds in seconds |
| Scanned PDF reads as empty | `python -m rag info` → is Tesseract found? |
| `404` on the model | Groq has no Llama models. Use `openai/gpt-oss-120b`. |
| Wrong answers | `python -m rag eval --retrieval-only` first — is it retrieval or the prompt? |

## One rule

If something is not working at 15:30, **cut it**. A smaller thing that demos
cleanly beats a bigger thing that fails in front of the panel.
