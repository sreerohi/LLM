# LLM
===Repository to explore various optimizations in LLMs===


A. Speculative Decoding is a technique usued to accelerate inference in LLMs using smaller LLMs without changing the output probability distribution of the tokens. The low latency of the smaller model combined with accuracy of the larger LLM makes this method very effective. 

In speculative_decoding.py, we have the implementation for the Speculative Decoding Algorithm .
Experiments:
1. Alpha (α) calculation
   <img width="1023" height="275" alt="image" src="https://github.com/user-attachments/assets/78133c86-f2cf-4c36-9ca1-2767aec5afc1" />

2. Plotting histogram of beta values for various time steps and observing acceptance/rejection
 <img width="1200" height="700" alt="image" src="https://github.com/user-attachments/assets/9ecf0fba-fdd8-410a-8c7d-5b5f8376996d" />

3. Exploring the relation between context and beta values
 <img width="735" height="579" alt="image" src="https://github.com/user-attachments/assets/d258b913-d8f4-449e-a3c0-efa634d02fa4" />


   
B. I have implemented autoregressive decoding with manual handling of KV Cache in autoregressive_decoding.py. 
There are two simple experiments with autoregressive decoding:
1. Experiment 1: Generating the same set of tokens is much faster with KV Cache.
2. Experiment 2: Smaller LLMs are faster than larger LLMs
Next steps: Implement latent KV  Cache and observe speedup 




References:
1.  Leviathan, Y., Kalman, M. & Matias, Y.. (2023). Fast Inference from Transformers via Speculative Decoding. Proceedings of the 40th International Conference on Machine Learning, in Proceedings of Machine Learning Research 202:19274-19286 Available from https://proceedings.mlr.press/v202/leviathan23a.html.
