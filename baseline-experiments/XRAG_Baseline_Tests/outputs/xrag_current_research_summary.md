# XRAG Project Current Research Summary

Generated: 2026-07-09 00:52

## Project framing

The project has two phases. First, test multilingual and cross-lingual RAG benchmarks such as XRAG to understand where current systems succeed and fail. Second, use those benchmark findings to build a multilingual RAG system with stronger retrieval, evidence filtering, and answer generation.

## Current research question

How does retrieved evidence quality affect multilingual RAG answer accuracy, language correctness, and unsupported-information errors?

## Completed benchmark work

- Built a 20-question Arabic/Spanish XRAG pilot.
- Tested GPT-5.4 under no-documents, support-only, and all-documents conditions.
- Tested BM25 retrieval, chunked multilingual E5 retrieval, and BM25+E5 hybrid retrieval.
- Ran an end-to-end BM25 Top-2 → GPT-5.4 generation condition.
- Manually scored answers with 0/1/2 correctness, refusal, language, unsupported information, and error category.

## GPT-5.4 condition comparison

| Condition | Score | Accuracy | Fully correct | Partial | Wrong | Unsupported |
|---|---:|---:|---:|---:|---:|---:|
| no_documents | 27/40 | 67.5% | 12 | 3 | 5 | 8 |
| support_only | 30/40 | 75.0% | 12 | 6 | 2 | 2 |
| all_documents | 26/40 | 65.0% | 10 | 6 | 4 | 9 |
| bm25_top2 | 31/40 | 77.5% | 14 | 3 | 3 | 6 |

## BM25 Top-2 generation summary

- BM25 Top-2 total score: 31/40 = 77.5%
- Complete support retrieved: 10/20
- Accuracy when complete support was retrieved: 85.0%
- Accuracy when support was incomplete: 70.0%
- Unsupported-information flags: 6/20
- Responses over 35 words: 2/20

BM25 Top-2 error categories:
- No error — fully correct: 14
- Incomplete answer: 3
- Distractor contamination: 1
- Wrong number or count: 1
- Wrong central answer: 1

## Retrieval baseline summaries

- BM25: 20 examples
  - Complete support @2: 0/20
  - Average Precision@2: 0.450
  - Average Recall@2: 0.625
- Chunked multilingual E5: 20 examples
  - Complete support @2: 0/20
  - Average Precision@2: 0.275
  - Average Recall@2: 0.350
- BM25+E5 hybrid: 20 examples
  - Complete support @2: 10/20
  - Average Precision@2: 0.000
  - Average Recall@2: 0.625

## Main findings so far

1. Evidence selection strongly affects final answer quality.
2. Providing all documents can hurt performance because distractors interfere with generation.
3. BM25 was the strongest retrieval baseline in the 20-question pilot.
4. Dense semantic retrieval did not automatically outperform keyword retrieval.
5. Complete-support coverage is more informative than MRR for cross-document QA.

## Next technical steps

1. Expand retrieval testing to a 50-question XRAG sample.
2. Evaluate BM25 at Top-1, Top-2, Top-3, and Top-4.
3. Analyze retrieval failure patterns by language, task, and question type.
4. Use those results to decide how many documents the first system should pass to the generator.
5. Build the first version of a multilingual RAG pipeline.