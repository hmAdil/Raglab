# Stage 1 hand-off (Member 2)

Top 3 configurations for Stage 2, in selection order (context_recall, then context_precision, then latency):
1. recursive__minilm__distilbert
2. recursive__bge__distilbert
3. recursive__bge__distilbert_bm25

Reading of the results (27 answerable questions, 32 configurations, mean over the other factors): the recursive chunker was best (recall 0.694), then fixed (0.676), sentence (0.652) and semantic (0.622), so the more elaborate semantic chunker was the worst. The embedder barely mattered (BGE 0.666 versus MiniLM 0.656). Adding BM25 through RRF raised mean recall from 0.624 to 0.698 but roughly doubled latency (16 ms to 32 ms). The largest surprise is ColBERT as the rescorer: dense-only ColBERT (0.583) was clearly worse than DistilBERT (0.664), and ColBERT only became competitive once BM25 was fused in (0.700). Several top configurations tie on recall (0.741), so the tie-break by precision and latency decided the top 3, and with 27 questions the differences between near-equal configurations are within noise.
