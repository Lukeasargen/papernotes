| Paper | Takeaways |
| :--- | :--- |
| 2023-07 SDXL Improving Latent Diffusion Models for High-Resolution Image Synthesis | more parameters 860M --> 2.6B. more conditioning inputs: concatenate 2 text embeddings channel wise, resolution, cropping, aspect ratio. new VAE trained with a larger batch size. (optional) more diffusion steps with refiner |

## 2023-07 SDXL Improving Latent Diffusion Models for High-Resolution Image Synthesis
- "3x larger UNet-backone compared to previous _Stable Diffusion_ models
- "additional condutioning techniques"
- "a separate diffusion-based refinement model which applies a noising-denoising [28] process to the latents produced by SDXL to improve the visual quality of its samples (Sec. 2.5)."
- "initial latents of size 128 x 128"
- "specialized high-resolution refinement model and apply SDEdit [28]"

![table 1](/figures/2023-07_SDXL_Improving_Latent_Diffusion_Models_for_High-Resolution_Image_Synthesis_Table_1.png)
- "For efficiency reasons, we omit the transformer block at the highest feature level, use 2 and 10 blocks at the lower levels, and remove the lowest level (8× downsampling) in the UNet altogether"
- "OpenCLIP ViT-bigG [19] in combination with CLIP ViT-L [34], where we concatenate the penultimate text encoder outputs along the channel-axis [1]. Besides using cross-attention layers to condition the model on the text-input, we follow [30] and additionally condition the model on the pooled text embedding from the OpenCLIP model."
- Conditioning the Model on Image Size "shortcoming of the LDM paradigm [38] is the fact that training a model requires a minimal image size, due to its two-stage architecture" "For this particular choice of data, discarding all samples below our pretraining resolution of 256^2 pixels would lead to a significant 39% of discarded data." "Instead, we propose to condition the UNet model on the original image resolution, which is trivially available during training. In particular, we provide the original (i.e., before any rescaling) height and width of the images as an additional conditioning to the model csize = (horiginal, woriginal). Each component is independently embedded using a Fourier feature encoding, and these encodings are concatenated into a single vector that we feed into the model by adding it to the timestep embedding [5]." "the effects of the size conditioning are less clearly visible after the subsequent multi-aspect (ratio) finetuning which we use for our final SDXL model."
- Conditioning the Model on Cropping Parameters "typical processing pipeline is to (i) resize an image such that the shortest size matches the desired target size, followed by (ii) randomly cropping the image along the longer axis" "During dataloading, we uniformly sample crop coordinates ctop and cleft (integers specifying the amount of pixels cropped from the top-left corner along the height and width axes, respectively) and feed them into the model as conditioning parameters via Fourier feature embeddings, similar to the size conditioning described above."
- "Given that in our experience large scale datasets are, on average, object-centric, we set (ctop, cleft) = (0, 0) during inference and thereby obtain object-centered samples from the trained model."
- Multi-Aspect Training "We follow common practice [31] and partition the data into buckets of different aspect ratios, where we keep the pixel count as close to 10242 pixels as possibly, varying height and width accordingly in multiples of 64. A full list of all aspect ratios used for training is provided in App. I." "the model receives the bucket size (or, target size) as a conditioning, represented as a tuple of integers car = (htgt,wtgt) which are embedded into a Fourier space in analogy to the size- and crop-conditionings described above"

![app I](/figures/2023-07_SDXL_Improving_Latent_Diffusion_Models_for_High-Resolution_Image_Synthesis_App_I.png)
- Improved Autoencoder "we can improve local, high-frequency details in generated images by improving the autoencoder." "we train the same autoencoder architecture used for the original Stable Diffusion at a larger batch-size (256 vs 9) and additionally track the weights with an exponential moving average."
- Refinement Stage "During inference, we render latents from the base SDXL, and directly diffuse and denoise them in latent space with the refinement model (see Fig. 1), using the same text input." "We note that this step is optional, but improves sample quality for detailed backgrounds and human faces"

**Takeaways**
- more parameters 860M --> 2.6B
- more conditioning inputs: concatenate 2 text embeddings channel wise, resolution, cropping, aspect ratio
- new VAE trained with a larger batch size
- (optional) more diffusion steps with refiner

