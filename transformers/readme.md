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

## 2018-09 Adaptive Input Representations for Neural Language Modeling
- [paperswithcode](https://paperswithcode.com/paper/adaptive-input-representations-for-neural)
- wikitext-103 18.7 perplexity
- adaptive input embeddings, "factorization assigns more capacity to frequent words and reduces the capacity for less frequent words with the benefit of reducing overfitting to rare words"
- "We show that variable-sized input embeddings can perform better than fixed sized embeddings. Furthermore, this also enables weight sharing with an adaptive softmax output layer."
- "When presented with a number of input words, the adaptive input embedding layer partitions the words into the various clusters, performs separate lookups in the embedding tables and then projects to dimension d, followed by concatenating the embeddings in the original order."
- "When the output layer is an adaptive softmax with the same partition ... then we can tie the weights"



## 2019-08 Adaptive Attention Span in Transformers
- [paperswithcode](https://paperswithcode.com/paper/adaptive-attention-span-in-transformers)


## 2020-04 Improving Transformer Models by Reordering their Sublayers
- [paperswithcode](https://paperswithcode.com/paper/improving-transformer-models-by-reordering)


## 2019-01 Transformer-XL Attentive Language Models Beyond a Fixed-Length Context

