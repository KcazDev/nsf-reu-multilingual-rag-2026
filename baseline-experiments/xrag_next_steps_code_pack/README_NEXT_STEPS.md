# XRAG Next Steps Code Pack

Place these files into your project folder:

```text
C:\Users\Ekalk\Desktop\NSF_REU_2026\Project\XRAG_Baseline_Tests
```

The Python files should go inside your existing `src` folder.

## Files included

| File | Purpose |
|---|---|
| `src/build_research_summary.py` | Creates a clean written research summary from your completed pilot results. |
| `src/create_xrag_eval_sample.py` | Creates a larger 50-question XRAG sample for the next retrieval experiment. |
| `src/evaluate_bm25_topk.py` | Evaluates BM25 retrieval at Top-1, Top-2, Top-3, and Top-4. |
| `src/analyze_bm25_topk.py` | Analyzes where BM25 succeeds/fails and recommends the best Top-K setting. |
| `src/run_rag_pipeline_v1.py` | First version of your own multilingual RAG pipeline using BM25 retrieval and optional GPT generation. |
| `run_next_steps.ps1` | Optional PowerShell runner for the non-API steps. |
| `requirements_next_steps.txt` | Package checklist. |

## Recommended order

From the project root, run:

```powershell
python src/build_research_summary.py
python src/create_xrag_eval_sample.py --sample-size 50 --languages ar es
python src/evaluate_bm25_topk.py --sample-file data/xrag_eval_sample_50.jsonl
python src/analyze_bm25_topk.py --results-file outputs/bm25_topk_eval_50.csv
python src/run_rag_pipeline_v1.py --sample-file data/xrag_eval_sample_50.jsonl --max-examples 3 --top-k 3
```

The final command previews your system pipeline without making API calls.

To generate answers with the OpenAI API later:

```powershell
python src/run_rag_pipeline_v1.py --sample-file data/xrag_eval_sample_50.jsonl --max-examples 1 --top-k 3 --generate
```

Only use `--generate` when you are ready to spend API credits.
