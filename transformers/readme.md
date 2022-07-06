## 2017-06 Efficient softmax approximation for GPUs
- [paperswithcode](https://paperswithcode.com/paper/efficient-softmax-approximation-for-gpus)
- logits complexity is linear with the size of the vocab
- word classes are build to minimize the computation time
- distribution of words follows a Zipf law
- small number of clusters (2 to 5), slightly better perplexity
- LSTM, one layer d=512, l2=1e-6
- word embedding is 256
- Adagrad, 20 steps BPTT, lr=0.1, 5 epochs, grad clip norm to 1, batch size 128 or 64
- used in Transformer XL
