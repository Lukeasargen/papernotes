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
- discriminator=critic. critic outputs linear value. gradients are better, but lipschitz continuous is a requirement. clamp the weights into a 0.01 box. don't use an optimizer with momentum.
