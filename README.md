# LLM
===Repository to explore various optimizations in LLMs===

A. In speculative_decoding.py, we have the implementation for the Speculative Decoding Algorithm .
Experiments:
1. Alpha (α) calculation
2. Plotting histogram of beta values for various time steps and observing acceptance/rejection
3. <img width="2968" height="1768" alt="image" src="https://github.com/user-attachments/assets/9ecf0fba-fdd8-410a-8c7d-5b5f8376996d" />

4. Exploring the relation between context and beta values

   
B. I have implemented autoregressive decoding with manual handling of KV Cache in autoregressive_decoding.py. 
There are two simple experiments with autoregressive decoding:
1. Experiment 1: Generating the same set of tokens is much faster with KV Cache.
2. Experiment 2: Smaller LLMs are faster than larger LLMs
Next steps: Implement latent KV  Cache and observe speedup 




References:
1.  Leviathan, Y., Kalman, M. & Matias, Y.. (2023). Fast Inference from Transformers via Speculative Decoding. Proceedings of the 40th International Conference on Machine Learning, in Proceedings of Machine Learning Research 202:19274-19286 Available from https://proceedings.mlr.press/v202/leviathan23a.html.
