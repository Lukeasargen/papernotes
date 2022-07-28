
## 2016-11 Categorical Reparameterization with Gumbel-Softmax
- [paperswithcode](https://paperswithcode.com/method/gumbel-softmax)
- "We introduce Gumbel-Softmax, a continuous distribution on the simplex that can approximate categorical samples, and whose parameter gradients can be easily computed via the reparameterization trick."
- "The Gumbel-Softmax distribution interpolates between discrete one-hot-encoded categorical distributions and continuous categorical densities." "Samples from Gumbel-Softmax distributions are identical to samples from a categorical distribution as T->0. At higher temperatures, Gumbel-Softmax samples are no longer one-hot, and become uniform as T->inf."
- "For learning, there is a tradeoff between small temperatures, where samples are close to one-hot but the variance of the gradients is large, and large temperatures, where samples are smooth but the variance of the gradients is small (Figure 1). In practice, we start at a high temperature and anneal to a small but non-zero temperature."
- "If T is a learned parameter (rather than annealed via a fixed schedule), this scheme can be interpreted as entropy regularization (Szegedy et al., 2015; Pereyra et al., 2016), where the Gumbel-Softmax distribution can adaptively adjust the “confidence” of proposed samples during the training process."
- Straight-Through (ST) Gumbel Estimator "we discretize y using argmax but use our continuous approximation in the backward pass by approximating"


## 2018-09 Adaptive Input Representations for Neural Language Modeling
- [paperswithcode](https://paperswithcode.com/paper/adaptive-input-representations-for-neural)
- wikitext-103 18.7 perplexity
- adaptive input embeddings, "factorization assigns more capacity to frequent words and reduces the capacity for less frequent words with the benefit of reducing overfitting to rare words"
- "We show that variable-sized input embeddings can perform better than fixed sized embeddings. Furthermore, this also enables weight sharing with an adaptive softmax output layer."
- "When presented with a number of input words, the adaptive input embedding layer partitions the words into the various clusters, performs separate lookups in the embedding tables and then projects to dimension d, followed by concatenating the embeddings in the original order."
- "When the output layer is an adaptive softmax with the same partition ... then we can tie the weights"
- Decoder only transformer, 16 layer, 16 heads, 1024 embedding size, 4096 FF size, Relu in FF
- attention drop 0.1 for BILLION WORD, increase to 0.3 for wikitext-103
- dropout 0.3 on embeddings
- 512 window
    - BILLION WORD 32 GPUs, 2048 batch
    - wikitext-103 8 GPUs, 4096 batch, 2 gradient accumulate steps
- BPE encoding, 32K merges, 1024 size embedding, "final evaluation is in terms word-level perplexity" "product of the sub-word units"
- SGD, nesterov, momentum=0.9, clip gradient norm to 0.1
- LR wamup, 1e-7 to 1 for 16k steps, cosine annealed for C cycles, cycle length doubles, reduce LR max by M, initial minimum LR is 1r-5
    - BILLION WORD, 975K updates, C=3 cycles (1st is 137K steps), M=0.6
    - wikitext-103, 286K steps, C=4 (1st is 18K steps), M=0.7
- trained at 16fp on V100
- wikitext-103, 18.7 ppl, 3072 window, 2560 context, score the last 512
- PPL results
    - "Adaptive input representations with tied input and output layers (ADP-T) achieve the highest accuracy at the same speed as the BPE models which have a very small vocabulary (33K versus 260K)."
    - "Fixed word embeddings perform least well (SM)."
    - "Sub-word units are fast to train and perform better than word models with fixed sized embeddings."
    - "For ASM, we found that reducing the dimension of the input word embeddings to 64 on WIKITEXT-103 results in better accuracy (Appendix A)."
- Rare words
    - "Regularization is more important on WIKITEXT-103 while models for BILLION WORD benefit from additional capacity."
    - "Tying weights helps all models on rare words, likely because of regularization effects."
    - "BPE and BPE-T perform poorly on rare words because probabilities are a product of several sub-word units."
    - "ADP-T performs best across all frequency ranges."
![table 5](/figures/2018_09_Adaptive_Input_Representations_for_Neural_Language_Modeling_Table_5.png)
- "Inference context is the number of tokens that are provided at evaluation before any tokens are scored."
- "Simply increasing the training block size from 512 to 3072 results in an improvement of nearly 1.2 perplexity with no inference context window. Increasing the context size at inference time results in an improvement of 0.6 perplexity for the largest training block size."
- Adaptive vs Full softmax
    - "We add dropout to the output of the first projection for all clusters, except for the head."
    - "This change enables the adaptive softmax to outperform a standard softmax over fixed size output word embeddings on WIKITEXT-103 (Table 6)."
    - "we found that adding dropout in this way is not helpful for larger datasets such as BILLION WORD."
    - "It may be possible to achieve better results by tuning dropout for each band of the tail and we leave this for future work."
- "Table 7 shows that reducing the capacity of fixed size word input embddings is beneficial on WIKITEXT-103."
- "We also experimented with sharing the head projection but found this to perform less well than not sharing it."

**Takeaways**
- 16K lr warmup steps with SGD up to lr=1, clip norm at 0.1
- BPE is poor for the perplexity at the word level, no comment on token perplexity
- adaptive softmax tied is the best, don't tie the projections in the head or tail
    - added dropout after the projection the tail, to the head, better for small datasets
- reduce the capacity of the embeddings for small datasets


## 2017-06 Efficient softmax approximation for GPUs
- [paperswithcode](https://paperswithcode.com/paper/efficient-softmax-approximation-for-gpus)
- logits complexity is linear with the size of the vocab
- "word classes are build to minimize the computation time"
- distribution of words follows a Zipf law
- small number of clusters (2 to 5), slightly better perplexity
- LSTM, one layer d=512, l2=1e-6
- word embedding is 256
- Adagrad, 20 steps BPTT, lr=0.1, 5 epochs, grad clip norm to 1, batch size 128 or 64
- used in Transformer XL

**Takeaways**
- adaptive softmax = less parameters, less memory, slightly better perplexity
