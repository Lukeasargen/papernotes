| Paper | Takeaways |
| :--- | :--- |
|2019-08 LXMERT: Learning Cross-Modality Encoder Representations from Transformers | More pretraining tasks empirically improves the validation metrics. However, you have to balance the pretraining task schedules based on the "difficulty" or "complexity" of the tasks. Balance the tasks complexity by starting and stopping the loss based on the lr. |
| 2020-05 Adaptive Transformers for Learning Multimodal Representations | These shallower networks are not as resilient to LayerDrop as deep networks. Adaptive span and adpative sparsity can work together. However, it seems that adaptave span works best with dense attention. Text can handle more sparsity than images. Images tended towards dense attention. |

## 2019-08 LXMERT: Learning Cross-Modality Encoder Representations from Transformers
- "multi-modality pre-training allows our model to infer masked features either from the visible elements in the same modality, or from aligned components in the other modality. In this way, it helps build both intra-modality and cross-modality relationships."
- "word-level sentence embeddings and object-level image embeddings"
- WordPiece tokenizer, absolute position embeddings added to input
- Faster R-CNN, pre-trained on Visual Genome
    - "We do not fine-tune the Faster R-CNN detector and freeze it as a feature extractor."
    - "keep 36 objects for each image"
- "We also show that loading BERT parameters into LXMERT will do harm to the pre-training procedure in Sec. 5.1 since BERT can perform relatively well in the language modality without learning these cross modality connections."
    - Language Task: Masked Cross-Modality LM: "randomly masked with a probability of 0.15 and the model is asked to predict these masked words"
    - Vision Task: Masked Object Prediction: "(i.e., masking RoI features with zeros) with a probability of 0.15 and asking the model to predict proprieties of these masked objects"
        - "RoI-Feature Regression regresses the object RoI feature with L2 loss"
        - "Detected-Label Classification learns the labels of masked objects with cross-entropy loss"
    - "Cross-Modality Matching For each sentence, with a probability of 0.5, we replace it with a mismatched sentence. Then, we train a classifier to predict whether an image and a sentence match each other."
    - Image Question Answering (QA)
        - "For the image QA pre-training tasks, we create a joint answer table with 9500 answer candidates which roughly cover 90% questions in all three image QA datasets."
- "We add these losses with equal weights."
- 9.18M image-and-sentence pairs on 180K distinct images
- Adam, lr linear decay, peak lr 1e-4, 670k steps (20 epochs), batch 256
    - "We only pre-train with image QA task (see Sec. 3.1.3) for the last 10 epochs, because this task converges faster and empirically needs a smaller learning rate."

**Takeaways**
- More pretraining tasks empirically improves the validation metrics. However, you have to balance the pretraining task schedules based on the "difficulty" or "complexity" of the tasks. Balance the tasks complexity by starting and stopping the loss based on the lr.

**Questions**
- To balance the losses for each task, could you weight the task by the std dev of the gradients? This is substitute for the noise of the predictions, which can be interpreted as model confidence in the task outputs. Downside is that the gradient statistics are dependent on batch size, label size, etc.

## 2020-05 Adaptive Transformers for Learning Multimodal Representations
- "Adaptive methods enforce the network to learn parameters such that their behavior changes as per the complexity of the input sequence as perceived by the neural network."
    - Adaptive attention span - each head learns the context size
    - Adaptive attention sparsity - entmax, sparse attention wieghts by assigning zero weights
    - LayerDrop - train by randomly dropping layers, prune the entire layer at inference
- LXMERT model - 9 layer text encoder, 5 layer image encoder, 5 layer multimodal encoder
- "The network used has been pre-trained on four objectives: Masked Cross Modality LM, Masked Object Prediction, Cross Modality Matching, and Image Question Answering"
- "Requiring a larger context size is indicative of the complexity of the sequences." "intermediate layers responsible for forming complex representations increase their spans" "Self attention also requires a high span when attending to visual features in the cross-modality encoder. This observation shows that visual sequences are perceived as a more complex input to process than a language input in the cross-modality encoder."
- "For dealing with language modality, self-attention favors mostly sparse mapping of attention weights in intermediate layers."
- "When vision modality is involved, heads that preferred sparse mapping initially are converging towards denser mapping, indicating that this representation of attention weights is preferred. We also observe that when both modalities are involved, the network prefers, even more, denser weight distribution. This observation shows that vision modality is given more preference (partly due to perceived complexity) over language inputs to process the sequence."
- [Author implementation](https://github.com/prajjwal1/adaptive_transformer)

**Takeaways**
- These shallower networks are not as resilient to LayerDrop as deep networks.
- Adaptive span and adpative sparsity can work together. However, it seems that adaptave span works best with dense attention.
- Text can handle more sparsity than images. Images tended towards dense attention.
