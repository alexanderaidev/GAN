# GAN

# 1: Discriminator
  A simple CNN that classifies images with outputs ranging between 0 and 1.

# 2: Generator:
  A generator that uses deconvolution layers to generate images.
  Special characteristics:
  - Three convolutions per upsampling step.
  - The first convolution uses BatchNorm and Noise Injection.
  - The second and third convolutions do not use normalization or noise injection, to avoid excessive noise and achieve better feature stabilization.
    
# 3: Training-File:
  This script allows you to train your GAN using both models.
  Everything is explained in detail through comments in the code.

#
**⚠️ Note:**  
  Replace every occurrence of `Your path here` in the code with your own file or folder paths, then you’re all set.

#

**⚠️ Work in Progress:**  
 - Working on a heuristic-based algorithm to dynamically turn discriminator training on and off based on past performance.

