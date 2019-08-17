# Confident Classifer

This approach follows the work of [Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples](https://arxiv.org/abs/1711.09325), where this method jointly trains both classification and generative neural networks for out-of-distribution tasks.

In this implementation, the classifier is an [Efficient-Net](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html) and the model used to generate out-of-distribution samples on the boundary is [DCGAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

#### Data processing

- This DCGAN accepts images of 128 x 128, so images were resized and center cropped.
- Training images that contained more than 0.5% black pixels are cropped 
- Training images were randomly flipped on both axes
- The usual pixel normalization was carried out

#### Training

Images were sampled from the dataloader using a `WeightedRandomSampler` to account for very uneven class sizes. `Adam` optimizer was used for all models, and learning rate was kept constant at `3e-4`.

Important GAN parameters:

- `nz = 100` (latent dimension)
- `ngf = 128`
- `ndf = int(ngf/4)`
- `beta = 0.1` (weight of KL loss term)

#### Future improvements

- Add white noise to discriminator input and other [tricks](https://github.com/soumith/ganhacks/blob/master/README.md)
- Retain a larger image size: substitute DCGAN for a [Progressive GAN](https://pytorch.org/hub/facebookresearch_pytorch-gan-zoo_pgan/)
- Train longer with annealing learning rate 





