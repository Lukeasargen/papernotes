
## 2018-04 Training Tips for the Transformer Model
- WIP

## 2020-12 MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers
- "We generalize and simplify deep self-attention distillation in MINILM by introducing multihead self-attention relation distillation, which brings more fine-grained self-attention knowledge and allows more flexibility for the number of student’s attention heads."
- "Taking query vectors as an example, in order to obtain queries of multiple relation heads, we first concatenate queries of different attention heads and then split the concatenated vector based on the desired number of relation heads. The same operation is also performed on keys and values."

*This doesn't make any sense to me. Concatenation and splitting will mix the heads in the distillation. Also, their teacher-student pairs still have the same number of attention heads.*

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
- Teacher: BERT_BASE. L12 E768 12 heads. 109M params
    - Students need to have 12 attention heads
- 30522 vocab, 512 max length
- Adam (0.9, 0.999), 0.1 dropout, weight decay 1e-2
- Student: 6L 768E, 1024 batch, 5e-4 lr, 400k steps
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
