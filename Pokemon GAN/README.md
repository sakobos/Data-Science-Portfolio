## Project Description <br>
The goal of this project is to build a Generative Adversarial Network to create images of Pokemon. <br>

## Data <br>
The image dataset was acquired from Kaggle user: HarshitDwivedi and can be found at: <br>
https://www.kaggle.com/datasets/thedagger/pokemon-generation-one <br>
The dataset contains a total of 6,820 images of the original 151 Pokemon taken from various productions of the images (trading cards, video games, TV series, etc.)
<br>
## Model Structure <br>
Currently (as of 03/21/2024) using a DCGAN structure almost identical (unintentionally) to the PyTorch DCGAN Faces Tutorial: <br>
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html <br>
<br>
## Results <br>
GAN currently does not produce good results. Images are entirely black, leading me to believe there is difficulty coming from the generation of the images. <br>
The first attempt to remedy the generation issue was to focus on creating images of just one Pokemon, a personal favorite, Charizard. <br>
That did not help the GAN so other avenues will be explored: <br>
- altering the structure of the GAN (specifically the generator).
- performing more transformations on the single Pokemon set to have more images to work with (potentially implement Differential Augmentation)
- could reintroduce the entire set of images and implement the same transformations on all 151 Pokemon species to have a significantly larger quantity of images for the generator to learn from. <br>

I do plan on moving the GAN over to my GPU.
