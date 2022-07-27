
## 2019-11 Improving Transformer Models by Reordering their Sublayers
- [paperswithcode](https://paperswithcode.com/paper/improving-transformer-models-by-reordering)
- Sandwich transformers
- WIP

## 2020-09 Pay Attention when Required
- [paperswithcode](https://paperswithcode.com/paper/pay-attention-when-required)
- Pay Attention when Required Trans former (or PAR Transformer)
- "We start with the intuition that attention blocks provides context meaning while being comparatively more expensive, and feedforward blocks provide content meaning."
- search on Transformer-XL
    1. "self-attention layers are necessary only among the former two-thirds layers of the network"
    2. "The total number of layers to self-attention layers ratio of p:1 is sufficient, with p=5 being optimal for Transformer-XL"
- "Sandwich transformers (Press et al., 2020), also keeps an equal number of self-attention and feed forward blocks but are designed using a sandwich coefficien instead of having a simple k interleaved design pattern. They have the firs k sublayers consisting of self-attention, the last sub k layers consisting of feed forward layers with both sandwiched between the classic interleaving pat tern of self-attention and feed forward blocks. This design pattern was found by conducting a series of random search experiments with constraints to keep the number of parameters constant."
- "identity block, feed forward block and self-attention block" "probability distribution computed by a Gumbel Softmax function" "Since the output at each layer is a linear combination of individual search blocks in that layer, the search cost is linear with respect to the number of blocks in the search space."
- "search also consists of training only one supernet consisting of all the search blocks"
- WikiText-103, L32, batch 128
    - architecture params: lr 1e-2, wd 5e-4
    - weight params: lr 1e-2, 2d 1e-4
- freeze architecture params for 10k steps
    - update architecture params for 20% of epoch for 40k steps
- stop training at architecture convergence, <75% of blocks don't change
- From 6 random seeds
    - ratio of layers:MSA is higher than 2:1. it's around 4:1. models use less MSA
    - most MSA are in the lower 2/3 of the model, with <1 MSA in the final 1/3
- PAR design rules
    1. MSA are placed uniformly in the lower 2/3 of the model
    2. Choose the number of MSA by the ratio of layers:MSA or p:1, where p>2.
- Choosen model is L32 p=5
- Latency is improved becuase the complexity of FFN is lower than MSA, particularly for more token inputs

**Takeaways**
- Train a "Supernet" for the architecture search. Linear search time for number of architecture components in each stage.
- MSA is best in the early layers.

## 2018-04 Training Tips for the Transformer Model
- "we use the case-insensitive sacréBLEU which uses a fixed tokenization"
- "In all cases, we plot the case-insensitive BLEU score against the wall-clock time in hours"
- word-piece tokenization. "shared source and target (English and Czech) subword vocabulary of size 32k" "average number of subwords per (space-delimited) word is about 1.5"
- "computation speed decreases with increasing batch size because not all operations in GPU are fully batch-parallelizable"
![table 2](/figures/2018-04_Training_Tips_for_the_Transformer_Model_Table_2.png)
- "Prefer the BIG over the BASE model if you plan to train longer than one day" "With less memory you should benchmark BIG and BASE with the maximum possible batch size."
- "Batch size should be set as high as possible"

**Takeaways**
- Use a bigger model. Use the largest batch size possible

## 2020-12 MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers
- "We generalize and simplify deep self-attention distillation in MINILM by introducing multihead self-attention relation distillation, which brings more fine-grained self-attention knowledge and allows more flexibility for the number of student’s attention heads."
- "Taking query vectors as an example, in order to obtain queries of multiple relation heads, we first concatenate queries of different attention heads and then split the concatenated vector based on the desired number of relation heads. The same operation is also performed on keys and values."

*For the most part they achieve identical results with half the layers AND half the hidden size (~0.25x # of params and ~2.7x speedip). Impressive and useful.*

**Takeaways**
- Attention distillation of query, key, and value relations separately. No longer restricted to the same number of attention heads by concatenating and spliting the queries, keys, and values.

## 2020-02 MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers
- [huggingface](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased)
- "we propose distilling the self-attention module of the last Transformer layer of the teacher"
![figure 1](/figures/2020_02_MiniLM_Deep_Self-Attention_Distillation_for_Task-Agnostic_Compression_of_Pre-Trained_Transformers_Figure_1.png)
- "we introduce the scaled dot-product between values in the self-attention module as the new deep self-attention knowledge, in addition to the attention distributions (i.e., the scaled dot-product of queries and keys)"
- "The vector representations of input tokens are computed via summing the corresponding token embedding, absolute position embedding, and segment embedding."
- QKV projections have the same dim
- 3.1. Self-Attention Distribution Transfer: "we minimize the KL-divergence between the self-attention distributions of ... the attention distributions of the last Transformer layer for the teacher and student"
- 3.2. Self-Attention Value-Relation Transfer: "scaled dot-product converts vectors of different hidden dimensions into the relation matrices with the same size"
- 3.3. Teacher Assistant: "For smaller students, we first distill the teacher into a teacher assistant ... the assistant model is then used as the teacher to guide the training of the final student."
- Teacher: BERT_BASE. L12 H768 12 heads. 109M params
    - Students need to have 12 attention heads
- 30522 vocab, 512 max length
- Adam (0.9, 0.999), 0.1 dropout, weight decay 1e-2
- Student: L6 H768, 1024 batch, 5e-4 lr, 400k steps
    - Other students 256 batch, 3e-4 lr
    - L12, H384, Adam (0.9, 0.98), 2048 batch, 6e-4 lr, 400k steps
    - L6, H384, 512 batch, 4e-4 lr
- LR linear warmup, 4K steps
- "We have also tried to transfer the relation between hidden states. But we find the performance of student models are unstable for different teacher models."
- [Example implementation](https://github.com/joanaapa/Distillation-DNABERT-Promoter/blob/f4c983b46448f8cea10bdac0a5c31effafe03ce1/src/transformers/modeling_minilm.py#L262)
    - Rebuilds the BertModel in huggingface with a new attention layer that outputs the queires, keys, and values
    - During [loss calculation](https://github.com/joanaapa/Distillation-DNABERT-Promoter/blob/f4c983b46448f8cea10bdac0a5c31effafe03ce1/distiller.py#L461) joanaapa uses nn.KLDivLoss on F.log_softmax. Not sure why, my guess is the log softmax is more stable and has better gradients

**Takeaways**
- BERT based. WordPiece tokenizer. Sum embeddings at input only.
- Distill last attention layer only. minimize KL on QK attention maps and value dot product. Models must have same number of heads in the last layer.
