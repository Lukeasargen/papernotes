## 2019-09 Reducing Transformer Depth on Demand with Structured Dropout
- [paperswithcode](https://paperswithcode.com/paper/reducing-transformer-depth-on-demand-with-1)
- "dropping layers during training can regularize and reduce the training time of very deep convolutional networks. In contrast, we focus on pruning"
- "we take advantage of the plasticity of neural networks to learn models that are resilient to random pruning, rather than learning the pruning itself"
- "In practice, we observe that networks are more robust to pruning than their expected ratio but higher pruning rates leads to better performance for smaller models."
- "We use a LayerDrop rate of p = 0.2 for all our experiments, but we recommend p = 0.5 to target very small inference time models."
- "Very deep Transformers are typically hard to train because of instability and memory usage, and they are prone to overfitting on a small dataset like Wikitext-103. LayerDrop regularizes the network, reduces the memory usage, and increases training stability as fewer layers are active at each forward pass."
- "We observe no large differences between dropping sub-layers and layers, possibly because we are working with relatively shallow networks. In theory, dropping sub-layers should perform better and we expect this to be the case with very deep Transformers."
- "the straight-forward strategy of selecting every other layer, is tough to beat. We find only marginal improvement can be gained by searching over the validation set for the best set of 8 layers to use and by learning which layers to drop."
- "The input and output layers of a network are the most important, as they process the input and project to the output vocabulary."

- https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/layer_drop.py
    - Keep all the model blocks in a nn.ModuleList and overload the __iter__ attribute to only yield p percent of the blocks

**Takeaways**
- Use LayerDrop if you intend to do layerwise structured pruning.

## 2019-09 Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
- Model parallel for the FFN along the columns of the weight. Does not require communication with the other columns since each is independent.
- Model parallel for the MSA along the head axis. Output layer is parallel along the rows.
- filter documents less than 128 tokens, use LSH to deduplicate jaccard >0.7
- init normal [0, 0.02], scale layers per residuals by 1/sqrt(2N) for N total layers
- Adam, wd 0.01, clip norm 1.0, dropout 0.1, activation checkpointing
- GPT-2: 50257 BPE vocab, context 1024, batch 512, 300k steps, lr 1.5e-4, linear warmup 3k steps, single cycle cosine decay, min lr 1e-5
- BERT: 30522 word-piece vocab, batch 1024, lr 1e-4, linear warmup 10k steps, 2M steps
- "hidden size per attention head is kept constant at 96"

**Takeaways**
- "model parallelism with only a few modifications"
- "careful attention to the placement of layer normalization in BERT-like models is critical to achieving increased accuracies as the model size increases"

## 2016-10 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- [paperswithcode](https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional)
- BERT: Bidirectional Encoder Representations from Transformers
- "bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers"
- BERTBASE (L=12, H=768, A=12, Total Parameters=110M)
    - Same size as GPT1
- BERTLARGE (L=24, H=1024, A=16, Total Parameters=340M)
- WordPiece embeddings, 30,000 vocab
- Append [CLS] token to the begining of a sequence. Use [SEP] token between sentences. Add different sentence embeddings for sentence A and B. Add embeddings to input token
- Pre training
    - MLM: "simply mask some percentage of the input tokens at random, and then predict those masked tokens. We refer to this procedure as a “masked LM” (MLM)" "mask 15% of all WordPiece tokens in each sequence at random" "mismatch between pre-training and fine-tuning, since the [MASK] token does not appear during fine-tuning" "we replace the i-th token with (1) the [MASK] token 80% of the time (2) a random token 10% of the time (3) the unchanged i-th token 10% of the time"
    - NSP: "when choosing the sentences A and B for each pretraining example, 50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext)"
- batch 256, length 512, 1M steps, Adam (0.9, 0.999), lr 1e-4, wd 1e-2, linear warmup 10k steps, dopout 0.1 all layers, gelu
-"To speed up pretraing in our experiments, we pre-train the model with sequence length of 128 for 90% of the steps. Then, we train the rest 10% of the steps of sequence of 512 to learn the positional embeddings."

**Takeaways**
- Bidirectional context is key for certain tasks. Masked Language Model (MLM) for pre-training. Then you finetune to perform many tasks.

**Questions**
- MLM schedule could improve training. Would progressively increasing the percentage of hidden tokens improve representations?

## 2019-01 Transformer-XL Attentive Language Models Beyond a Fixed-Length Context
- [paperswithcode](https://paperswithcode.com/paper/transformer-xl-attentive-language-models)
- "the model cannot capture any longer-term dependency beyond the predefined context length. In addition, the fixed-length segments are created by selecting a consecutive chunk of symbols without respecting the sentence or any other semantic boundary. Hence, the model lacks necessary contextual information needed to well predict the first few symbols, leading to inefficient optimization and inferior performance. We refer to this problem as context fragmentation."
- "we reuse the hidden states obtained in previous segments. The reused hidden states serve as memory for the current segment, which builds up a recurrent connection between the segments."
- weight tied output layer
- "the hidden state sequence computed for the previous segment is fixed and cached to be reused as an extended context when the model processes the next new segment"
- "With this recurrence mechanism applied to every two consecutive segments of a corpus, it essentially creates a segment-level recurrence in the hidden states. As a result, the effective context being utilized can go way beyond just two segments"
- "instead of incorporating bias statically into the initial embedding, one can inject the same information into the attention score of each layer."
- "it suffices to know the relative distance between each key vector kj and itself qi, i.e. i-j."
- "our relative positional embedding R adapts the sinusoid bias, a model trained on a memory of some certain length can automatically generalize to a memory several times longer during evaluation"
- [Author implementation](https://github.com/kimiyoung/transformer-xl)
- Pytorch https://github.com/sooftware/conformer/blob/main/conformer/attention.py#L26
- Pytorch https://github.com/oshindow/Transformer-Transducer/blob/master/tt/transformer/attention.py#L121

**Takeaways**
- Cache previous hidden states/memory; adds RNN like structure to the model. Memory length can be extended during evaluation.
- Relative positions encodings. Reparametrize so that the QK relations are separated into content and position biases. Use sinusoid relative embeddings to extend context beyond training length.

**Questions**
- Why cache the whole context? Model should decide what to keep.

## 2019-05 Adaptive Attention Span in Transformers
- [paperswithcode](https://paperswithcode.com/paper/adaptive-attention-span-in-transformers)
- relative position embeddings. added to the keys
- "for each input token, each attention head scales linearly in memory and time in the context size, or attention span"
- "Each attention head of a Transformer shares the same attention span S. This assumes that every head requires the same span to form its representation."
- "add a masking function to control for the span of the attention" "parametrized by a real value in z [0, S]" "add a l1 penalization on the parameters z"
- 8 heads, ReLU, all heads share the same learned position embeddings
    - small: L12 H512, ffn 2048 dropout 0.3, 600k steps (900k for S=8192)
    - large: L24 H768, ffn 4096 dropout 0.4, (250k text8, 150k enwik8), 20k steps with lr/10
- dynamic span starts with -4 bias to make intial spans small
- $\lambda$=2e-6, R=32
    - reduced to $\lambda$=0.5e-6 for S=8192 "because z was not growing longer than 4000"
- adagrad, btach64, fixed lr 0.07, 32k linear warmup steps, value clip 0.03, 512 training context
- "Interestingly, even with a limit on span sets to 8192, the average span is only 314."
- "We can see that the lowest 5 layers have the smallest possible attention span, which is R = 32 of the masking function." "Although there is a general tendency of higher layers having longer attention spans, it is not a simple monotonic function of the layer height."
- [Author implementation in pytorch](https://github.com/facebookresearch/adaptive-span)
- https://github.com/prajjwal1/fluence/blob/master/fluence/adaptive/span.py

**Takeaways**
- Learning the attention span does not degrade performance.
- Reduce memory and compute costs.

## 2019-11 Improving Transformer Models by Reordering their Sublayers
- [paperswithcode](https://paperswithcode.com/paper/improving-transformer-models-by-reordering)
- Sandwich transformers
- "Could ordering the sublayers in a different pattern lead to better performance?"
- "interleaving pattern of self-attention and feedforward sublayers (sfsfsf...)" "there is no reason to expect this particular pattern to be optimal"
- WikiText-103 word-level
- "First, we generate random transformer models while keeping the number of parameters constant." "a third of our random models outperformed the average baseline suggests that a better ordering than interleaving probably exists"
- L16, 16 heads, d=1024 hidden, ffn 4096
    - param counts. msa=4d^2. ffn=8d^2. ffn has 2x msa params
- "we generate 20 unbalanced transformer models by randomly selecting one sublayer at a time (either s or f with equal probability) until the parameter budget is exhausted."
    - "We do not observe any preference for more sublayers of one type over the other"
    - "we conclude that a balanced number of self-attention and feedforward sublayers seems to be a desirable property, though not a necessary one"
- "models that outperform the average baseline tend to have more self-attention s in the first (bottom) half of the network and more f in the second (top) half"
- Sandwich transformers
    - 2n sublayers in total (n of each type)
    - first k sublayers are purely self-attention
    - last k are feedforward sublayers
    - original interleaving pattern (sf) to fill the remaining 2(n-k) sublayers
    - "We refer to k as the transformer’s sandwich coefficient."

![figure 5](/figures/2019-11_Improving_Transformer_Models_by_Reordering_their_Sublayers_Figure_5.png)
- "This experiment indicates that a reordering pattern that benefits one particular task (language modeling) might not carry the same performance gains to another (machine translation). However, it also demonstrates the general robustness of transformer architectures to sublayer reordering"
- [Author implementation in pytorch](https://github.com/ofirpress/sandwich_transformer)
    - adaptive span: only trim the keys and values because you need queries for every token. chunk the trim into 64 token sizes to help with memory management

**Takeaways**
- The search did not favor MSA or FFN over the other. Use MSA first and FFN later.

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
![figure 1](/figures/2020-09_Pay_Attention_when_Required_Figure_1.png)
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
- [Example implementation in tf](https://github.com/Jmkernes/PAR-Transformer-XL/blob/main/par_model.py#L171)
    - Create a stochastic block that stores the weight of each branch
    - Use the Gumbel-Softmax to train the architecture weights. I'm pretty sure you just accumulate the gradients for a bunch of batches
    - [GIF by Jmkernes](https://raw.githubusercontent.com/jmkernes/PAR-Transformer-XL/main/movie.gif) shows how the branch weights update during training

**Takeaways**
- Train a "Supernet" for the architecture search. Linear search time for number of architecture components in each stage.
- MSA is best in the early layers.

**Questions**
- What other branches can be used? fourier transform maybe have useful interactions with sinusoidal position embeddings

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
- Distill the Q-Q, K-K, V-V relations of only ONE teacher layer into the last layer of a student model.
- Concatenate the QKV projections and split into multiple relation heads. Calculate the scaled dot product. Use the KL loss.

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
- [Example implementation in pytorch](https://github.com/joanaapa/Distillation-DNABERT-Promoter/blob/f4c983b46448f8cea10bdac0a5c31effafe03ce1/src/transformers/modeling_minilm.py#L262)
    - Rebuilds the BertModel in huggingface with a new attention layer that outputs the queires, keys, and values
    - During [loss calculation](https://github.com/joanaapa/Distillation-DNABERT-Promoter/blob/f4c983b46448f8cea10bdac0a5c31effafe03ce1/distiller.py#L461) joanaapa uses nn.KLDivLoss on F.log_softmax. Not sure why, my guess is the log softmax is more stable and has better gradients

**Takeaways**
- BERT based. WordPiece tokenizer. Sum embeddings at input only.
- Distill last attention layer only. minimize KL on QK attention maps and value dot product. Models must have same number of heads in the last layer.

**Questions**
- Does this work for masked attention in decoders?
- Is there an equivalent cross-attention relation?
