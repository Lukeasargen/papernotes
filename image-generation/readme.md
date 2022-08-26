| Paper | Model | Takeaways |
| :--- | :----: | :--- |
| 2017-01 Wasserstein GAN | WGAN | Earth-Mover (EM) distance or Wasserstein-1. discriminator=critic. critic outputs linear value. gradients are better, but lipschitz continuous is a requirement. clamp the weights into a 0.01 box. don't use an optimizer with momentum. |
| 2017-03 Improved Training of Wasserstein GANs| WGAN-GP | two-sided gradient norm penalty to enforce 1-Lipschtiz in the critic. optimizer with momentum works |
| 2016-05 Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network | SRGAN | augment MSE loss (content) with GAN loss. balancing these losses is still an issue |
| 2016-05 Generative Adversarial Text to Image Synthesis | |supervised pretraining of text encoder. not required, stills works end-to-end. use pretrained conv features. loss is a similarity between pairs of image-text for the encoders/classifiers. concatenate text features to noise in G and spatial features in D. add loss to D to learn both real/fake and image-text matching. interpolations in the text embeddings |
| 2016-06 Improved Techniques for Training GANs | | feature matching. minibatch discrimination. historical averaging. one-sided label smoothing, "smooth only the positive labels to $\alpha$." virtual batch normalization. semi-supervised learning by adding a "generated" prediction to a classifier. weight normalization in D. add Gaussian noise to activations in D. |

## 2016-06 Improved Techniques for Training GANs
- Nash equilibrium. Loss is a minimum of both the generator and discriminator.
- feature matching. "we train the generator to match the expected value of the features on an intermediate layer of the discriminator"
- minibatch discrimination. "modelling the closeness between examples in a minibatch"
- historical averaging. modify the cost with L2 distance with previous parameters
- one-sided label smoothing. "replaces the 0 and 1 targets for a classifier with smoothed values, like .9 or .1" "smooth only the positive labels to $\alpha$, leaving negative labels set to 0."
- virtual batch normalization. "Batch normalization ... causes the output of a neural network for an input example x to be highly dependent on several other inputs x' in the same minibatch" "each example x is normalized based on the statistics collected on a reference batch of examples that are chosen once and fixed at the start of training, and on x itself"
- inception score. "important to evaluate the metric on a large enough number of samples (i.e. 50k) as part of this metric measures diversity." _so the metric is dependent on number of samples. you can compare inception score if the number of samples is the same_
- "We can do semi-supervised learning with any standard classifier by simply adding samples from the GAN generator G to our data set, labeling them with a new “generated” class"
    - $L_{supervised}$ (the negative log probability of the label, given that the data is real)
    - $L_{unsupervised}$ which is in fact the standard GAN game-value
    - classes+1 outputs is over-parameterized. "standard supervised loss function of our original classifier with K classes, and our discriminator D is given by $ D(x) = \frac{Z(x)}{Z(x)+1}$, where $Z(x) = \Sigma_{k=1}^{K} exp[l_k(x)]$ ."
- "By having the discriminator classify the object shown in the image, we bias it to D develop an internal representation that puts emphasis on the same features humans emphasize." _sources?_
- "We use weight normalization [20] and add Gaussian noise to the output of each layer of the discriminator."

**Takeaways**
- feature matching. minibatch discrimination. historical averaging. one-sided label smoothing, "smooth only the positive labels to $\alpha$." virtual batch normalization. semi-supervised learning by adding a "generated" prediction to a classifier. weight normalization in D. add Gaussian noise to activations in D.

## 2016-05 Generative Adversarial Text to Image Synthesis
- "we aim to learn a mapping directly from words and characters to image pixels"
- "By conditioning both generator and discriminator on side information, we can naturally model this phenomenon since the discriminator network acts as a “smart” adaptive loss function."
- "character-level text encoder and class-conditional GAN"
- "The intuition here is that a text encoding should have a higher compatibility score with images of the correspondong class compared to any other class and vice-versa." "The resulting gradients are backpropagated through $\varphi$ to learn a discriminative text encoder."
- "for full generality and robustness to typos and large vocabulary, in this work we always used a hybrid characterlevel convolutional-recurrent network"
- generator G - sample noise prior, encode text query, affine to 128-dim, LeakyReLU, concatenate to noise vector
- discriminator D - separate afine to 128-dim, LeakyReLU, broadcast to 4x4 spatial dim and concatenate features, 1x1 conv, LeakyReLU, 4x4 conv produces the final score

![figure 2](/figures/2016-05_Generative_Adversarial_Text_to_Image_Synthesis_Figure_2.png)
- "In addition to the real / fake inputs to the discriminator during training, we add a third type of input consisting of real images with mismatched text, which the discriminator must learn to score as fake. By learning to optimize image / text matching in addition to the image realism, the discriminator can provide an additional signal to the generator."
- "we can generate a large amount of additional text embeddings by simply interpolating between embeddings of training set captions" "Because the interpolated embeddings are synthetic, the discriminator D does not have “real” corresponding image and text pairs to train on. However, D learns to predict whether image and text pairs match or not. Thus, if does a good job at this, then by satisfying D on interpolated text embeddings G can learn to fill in gaps on the data manifold in between training points."
- "For text features, we first pre-train a deep convolutional recurrent text encoder on structured joint embedding of text captions with 1,024-dimensional GoogLeNet image embedings" "pre-training the text encoder is not a requirement of our method and we include some end-to-end results in the supplement." "The reason for pre-training the text encoder was to increase the speed of training the other components for faster experimentation."
- Adam, lr 2-4, beta1=0.5, batch 64, 600 epochs
- "The basic GAN tends to have the most variety in flower morphology,while other methods tend to generate more class-consistent images."

![figure 2](/figures/2016-05_Generative_Adversarial_Text_to_Image_Synthesis_Figure_8.png)
- "A common property of all the results is the sharpness of the samples, similar to other GAN-based image synthesis models. We also observe di versity in the samples by simply drawing multiple noise vectors and using the same fixed text encoding."

**Takeaways**
- supervised pretraining of text encoder. not required, stills works end-to-end. use pretrained conv features. loss is a similarity between pairs of image-text for the encoders/classifiers
- concatenate text features to noise in G and spatial features in D. add loss to D to learn both real/fake and image-text matching.
- interpolations in the text embeddings

## 2016-05 Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
- "the ability of MSE (and PSNR) to capture perceptually relevant differences, such as high texture detail, is very limited as they are defined based on pixel-wise image differences"
- "single image super-resolution (SISR)"
- "enabling the network to learn the upscaling filters directly can further increase performance both in terms of accuracy and speed"
- "minimizing MSE encourages finding pixel-wise averages of plausible solutions which are typically overly-smooth and thus have poor perceptual quality"
- "GAN drives the reconstruction towards the natural image manifold producing perceptually more convincing solutions"
- "In training, $I^{LR}$ is obtained by applying a Gaussian filter to $I^{HR}$ followed by a downsampling operation with downsampling factor r."
- "we will specifically design a perceptual loss $l^{SR}$ as a weighted combination of several loss components that model distinct desirable characteristics of the recovered SR image."
- "We increase the resolution of the input image with two trained sub-pixel convolution layers"
- EQ 3: $l^{SR}$ = $l^{SR}_X$ + $10^{-3}l^{SR}_{Gen}$
- "We obtained the LR images by downsampling the HR images (BGR, C = 3) using bicubic kernel with downsampling factor r = 4. For each mini-batch we crop 16 random 96 x 96 HR sub images of distinct training images."
- "We scaled the range of the LR input images to [0, 1] and for the HR images to [-1, 1]."
- Adam with beta1=0.9m lr 1e-4

**Takeaways**
- augment MSE loss (content) with GAN loss. balancing these losses is still an issue

## 2017-03 Improved Training of Wasserstein GANs
- "We propose an alternative to clipping weights: penalize the norm of gradient of the critic with respect to its input"
- gradient penalty, WGAN-GP
- "WGAN requires that the discriminator (called the critic in that work) must lie within the space of 1-Lipschitz functions"
- "A differentiable function is 1-Lipschtiz if and only if it has gradients with norm at most 1 everywhere, so we consider directly constraining the gradient norm of the critic’s output with respect to its input."
- "Penalty coefficient All experiments in this paper use $\lambda$ = 10"
- "For WGAN-GP, we replace any batch normalization in the discriminator with layer normalization."
- "One advantage of our method over weight clipping is improved training speed and sample quality." "Using Adam further improves performance."
    - Adam (lr = .0001, $\beta_1$ = .5, $\beta_2$ = .9)

**Takeaways**
- two-sided gradient norm penalty to enforce 1-Lipschtiz in the critic. optimizer with momentum works

## 2017-01 Wasserstein GAN
- Purposes Earth-Mover (EM) distance or Wasserstein-1 for the WGAN
- Requires the function to be locally Lipschitz
- Wasserstein GAN, WGAN
- Kantorovich-Rubinstein duality
- EQ 2
    - W is the new loss function
    - f is 1-Lipschitz function
- EQ 3
    - f_w is a funciton on the distribution, so the discriminator
    - find w weights of the function to get the smallest maximum EM distance
- Theorem 3
    - it looks like f is just the output of the discriminator. there is no BCE loss
- f_w is called the critic
- "In order to have parameters w lie in a compact space, something simple we can do is clamp the weights to a fixed box (say W = [-0.01, 0.01]) after each gradient update." "Weight clipping is a clearly terrible way to enforce a Lipschitz constraint." "we stuck with weight clipping due to its simplicity and already good performance"
- **"The critic, however, can't saturate, and converges to a linear function that gives remarkably clean gradients everywhere.** The fact that we constrain the weights limits the possible growth of the function to be at most linear in different parts of the space, forcing the optimal critic to have this behaviour."

![figure 5](/figures/2017-01_Wasserstein_GAN_Figure_2.png)
- "training becomes unstable at times when one uses a momentum based optimizer such as Adam" "the loss for the critic is nonstationary, momentum based methods seemed to perform worse."
- "When the critic is trained to completion, it simply provides a loss to the generator that we can train as any other neural network. This tells us that we no longer need to balance generator and discriminator's capacity properly. The better the critic, the higher quality the gradients we use to train the generator."

**Takeaways**
- Earth-Mover (EM) distance or Wasserstein-1. discriminator=critic. critic outputs linear value. gradients are better, but lipschitz continuous is a requirement. clamp the weights into a 0.01 box. don't use an optimizer with momentum.
