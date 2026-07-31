python src/build_research_summary.py
python src/create_xrag_eval_sample.py --sample-size 50 --languages ar es
python src/evaluate_bm25_topk.py --sample-file data/xrag_eval_sample_50.jsonl
python src/analyze_bm25_topk.py --results-file outputs/bm25_topk_eval_50.csv
python src/run_rag_pipeline_v1.py --sample-file data/xrag_eval_sample_50.jsonl --max-examples 3 --top-k 3
