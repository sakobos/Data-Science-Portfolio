## Project Description <br>
The goal of this project is to build a Generative Adversarial Network to create images of Pokemon. <br>

## Data <br>
The image dataset was acquired from Kaggle user: HarshitDwivedi and can be found at: <br>
https://www.kaggle.com/datasets/thedagger/pokemon-generation-one <br>
The dataset contains 60 or more images of each of the original 151 Pokemon, resulting in over 10,000 total images.  
<br>
## Model Structure <br>
Currently (as of 03/21/2024) using a DCGAN structure almost identical (unintentionally) to the PyTorch DCGAN Faces Tutorial: <br>
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html <br>
<br>
## Results <br>
GAN currently does not produce good results. Images are essentially entirely black, leading me to believe there is difficulty coming from the generation of the images. <br>
The first attempt to remedy the generation issue was to focus on creating images of just one Pokemon, a personal favorite, Charizard. <br>
That did not help the GAN so other avenues will be explored: <br>
- altering the structure of the GAN (simplify the whole thing, or maybe reduce the image less)
- performing more transformations on the single Pokemon set to have more images to work with
- testing different parameters such as learning rate, epochs, etc.
- exploring the potential of differentiable augmentation

## Extra <br>
I am having difficulty getting the project to run on my GPU (M2 Macbook Air, MPS). If anyone that sees this can help it would be greatly appreciated. 

