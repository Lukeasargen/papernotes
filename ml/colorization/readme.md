| Paper | Takeaways |
| :--- | :--- |
| 2022-06 Neural Image Recolorization for Creative Domains | Output is RGB. Step 1 is train a exemplar based generator. Step 2 is learn the color patlette and illumination mapping into the latent space from step 1. content encoder uses positional normalization |
| 2021-06 Color2Embed Fast Exemplar-Based Image Colorization using Color Embeddings | Use transforms of GT to get Rrgb. Pass Rrgb into color encoder. Content encoder. U-Net like. Decoder has convolution weights modulated like StyleGAN2 |
| 2018-07 Deep Exemplar-based Colorization | Similarity sub-net to warp the reference colors to be more semantically aligned with the target grayscale. Colorization sub-net (U-net) to predict the ab channels for the target |
| 2018-03 Image Colorization with Generative Adversarial Networks | Lab color space. U-Net. Leaky-ReLU. "Batch-Norm is not applied on the first layer". Adam beta1=0.5. one sided label smoothing. imbalanced colors in datasets leads to objects colored with the most frequently seen colors. | 


## 2022-06 Neural Image Recolorization for Creative Domains
- ColorHouse
- "While the colors of a random scene may be arbitrarily distributed, creative works often adhere to a more structured palette." "we can think of color palettes as human-understandable parameterizations of this space"
- "our system both conditions on and generates full RGB images"
- "exemplar-based ColorHouse generator is trained in the first stage, shown on the left side of Fig 2, as a style transfer task"
- content encoder. "moment shortcut extracts the mean and standard deviation based on positional normalization for use in the later layers; the extracted information captures structural information in the input and this connection can boost the training performance while preserving the structure"
- "In step 2 of our approach we learn a mapping from vectors consisting of 5 palette colors and one illumination color to the latent color space learned in step 1."
- "The palette extractor consists of two 3 × 3 stride-1 convolutional layers. After each layer, we apply an adaptive pooling layer to downsample the features. In the end, we apply two linear layers to model interdependencies between channels and learn channel-wise feature responses adaptively"
- "we condition our palette-based generator on a target illumination color" "we use a differentiable network to predict the illumination"

**Takeaways**
- Output is RGB. Step 1 is train a exemplar based generator. Step 2 is learn the color patlette and illumination mapping into the latent space from step 1. content encoder uses positional normalization

## 2021-06 Color2Embed Fast Exemplar-Based Image Colorization using Color Embeddings
- CIE Lab "we directly extract the color embeddings in RGB color space while our reconstruction process is performed in Lab color space"
- Ec = color encoder. Es = color encoder. Gpffn = Progressive Feature Formalisation Network (PFFN)
    - Ec takes reference RGB and generatres color embeddings z
    - Es is the downsampling resblocks to extract intrmediate features
    - Gpffn - upsampling and Progressive Feature Formalisation Blocks (PFFB)
- "we generate Rrgb from ground truth by applying the Thin Plate Splines (TPS)" "Since Rrgb is essentially generated from GT, these processes guarantee the sufficient
color information to colorize TL"
- "we embed the feature modulation module from StyleGAN2 [21] into our PFFN." "After the modulation, we adopt the normalization procedure"
- smoothL1 & Perceptual loss (VGG relu5_2)
- "We resize all training images into size 256x256 with bilinear rescaling method"
- "During generating reference image, we add random Gaussian noise with mean 0 and variance $\sigma$ = 5."
- color dim 512. batch 64. Adam (beta1=0.9, beta2=0.99). lr 1e-4
    - perceptual loss is scaled by 0.1

**Takeaways**
- Use transforms of GT to get Rrgb. Pass Rrgb into color encoder.
- Content encoder. U-Net like. Decoder has convolution weights modulated like StyleGAN2

## 2018-07 Deep Exemplar-based Colorization
- "ill-conditioned and inherently ambiguous since there are potentially many colors that can be assigned to the gray pixels of an input image (e.g., leaves may be colored in green, yellow, or brown)"
- "color reference image similar to the grayscale image is given to facilitate the process" "the quality of the result depends heavily on the choice of reference"
- Similarity sub-net "semantic similarity between the reference and the target using a VGG-19 network pre-trained on the gray-scale image object recognition task"
- Colorization sub-net "multi-task learning to train two different branches" "1) Chrominance loss, which encourages the network to selectively propagate the correct reference colors for relevant patch/pixel, satisfying chrominance consistency; 2) Perceptual loss, which enforces a close match between the result and the true color image of high-level feature representations"
- "jointly learns faithful local colorization to a meaningful reference and plausible color prediction when a reliable reference is unavailable"
- "to measure the semantic relationship" "use a gray-VGG-19, trained on image classification tasks only using the luminance channel"
- "Our system uses the CIE Lab color space, which is perceptually linear."
- "Similarity sub-net computes the semantic similarities between the reference and the target, and outputs bidirectional similarity maps simT<->R. The Colorization sub-net takes simT<->R, TL and Rab as the input, and outputs the predicted ab channels of the target Pab, which are then combined with TL to get the colorized result PLab (PL = TL)"
- Similarity sub-net, TL and RL, extract feature pyramid, upsample to input resolution, cosine similarity
- Colorization sub-net, 13 channels, 2x 5 similarity features, TL, Rab
    - Chrominance branch. "propagate the correct reference colors" "we leverage the bidirectional mapping functions to reconstruct a ”fake” reference Tab' ground truth chrominance" smooth L1
    - Perceptual branch. "generate the predicted chrominance Pab" L2 fistance of feature maps
    - $\alpha$ is empirically set to 0.005
- U-net. conv-relu-bn, "dilated convolution layers with a factor of 2 are used in the 5th and 6th convolutional blocks" "down-sampling layers use convolution with stride 2, while all upsampling layers use deconvolution with stride 2" "final layer is a tanh layer"
- Adam. batch 256 (128 for Chrominance branch and 128 for Perceptual branch). init lr 1e-4, decay 0.1 every 3 epochs, 10 epochs
- "The ideal reference is expected to match the target image in both semantic content and photometric luminance." "features are pre-computed and stored in the database for the latter query"
- "We compress neural features with the common PCA-based compression [Babenko et al. 2014] to accelerate the search. The channels of feature fc6 are compressed from 4096 to 128 and the channels of features relu5_$ are reduced from 512 to 64 with practically negligible loss. After these dimensionality reductions, our refer ence retrieval can run in real-time."

**Takeaways**
- Use Similarity sub-net to warp the reference colors to be more semantically aligned with the target grayscale.
- Use the colorization sub-net (U-net) to predict the ab channels for the target

## 2018-03 Image Colorization with Generative Adversarial Networks
- "L* a* b* color space contains dedicated channel to depict the brightness of the image and the color information is fully encoded in the remaining two channels. As a result, this prevents any sudden variations in both color and brightness through small perturbations in intensity values that are experienced through RGB."
- "4x4 convolution layers with stride 2 for downsampling" "Leaky-ReLU activation function with the slope of 0.2" "4x4 transposed convolutional layer with stride 2 for upsampling" "tanh function for the last layer" "For discriminator D, we use similar architecture as the baselines contractive path"
- "minimize the Euclidean distance between predicted and ground truth averaged over all pixels"
- adam, lr 2e-4 for G and D, plateau schedule steps by 1/10
- One sided label smoothing, only positive labels to 0.9
- "Batch-Norm is not applied on the first layer of generator and discriminator and the last layer of the
generator as suggested by [5]."
- Reduce Adam beta1=0.5, reduces the momentum term to change the oscillation instability
- accuracy. "Any two pixels are considered to have the same color if their underlying color channels lie within some threshold distance $\epsilon$."
- "one drawback was that the GAN tends to colorize objects in colors that are most frequently seen" "many car images were colored red" "regions of images that have high fluctuations are frequently colored green"

**Takeaways**
- Lab color space. U-Net. Leaky-ReLU. "Batch-Norm is not applied on the first layer". Adam beta1=0.5. one sided label smoothing.
- imbalanced colors in datasets leads to objects colored with the most frequently seen colors
