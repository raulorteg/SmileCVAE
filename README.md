# Conditional Smiles! (SmileCVAE)

## About
Implementation of AE, VAE and CVAE. Trained CVAE on faces from  [UTKFace Dataset](https://susanqq.github.io/UTKFace/). With an (handmade) encoding of the Smile-strength degree to produce conditional generation of synthetic faces with a given smile degree.

## Installation
1. Clone the repository ``` git clone https://github.com/raulorteg/SmileCVAE```
2. Create virtual environment:
* Update pip ``` python -m pip install pip --upgrade ```
* Install ``` virtualenv ``` using pip ``` python -m pip install virtualenv ```
* Create Virtual environment ``` virtualenv SmileCVAE ```
* Activate Virtual environment (Mac OS/Linux: ``` source SmileCVAE/bin/activate ```, Windows: ``` SmileCVAE\Scripts\activate ```)
* (_Note: to deactivate environemt run ``` deactivate ```_)
3. Install requirements on the Virtual environment ``` python -m pip install -r requirements.txt ```
4. Create the directory structure ``` python setup_directories.py ```

## Usage
### With shell scripts files:
_Note: Shell scripts are created to easily train and obtain visualizations. In scripts/pc 
5 different scripts can be found, each correspond to calls to different python scripts, to run everything:_ 
1. Navigate to ```cd scripts/pc```
2. To run all scripts ``` sh runall.sh```

### Manually
1. Train the models (e.g ``` python cvae.py --n_epochs=1000 --lr=0.00002 --beta=0.5 --batch_size=8 --latent_size=20 --resize=50 ```)
2. Obtain synthetic samples from the latent space (e.g ``` python sample_prior.py --degree=0.6 --resize=50 --latent=20 ```)
3. Force smiles on people (e.g ``` python changing_smiles.py --degree=0.6 --resize=50 --latent=20 ```)
4. Inspect the axis-th dimension of the latent space (e.g ``` python sample_across_axis.py --axis=0 --resize=50 --latent=20 ```)

## Results
### Training
In the .gif below the reconstruction for a group of 32 faces from the dataset can be visualized for all epochs.
![Training](results/cvae_training.gif)

Below, the final reconstruction of the CVAE for 32 faces of the dataset side by side to those original 32 images, for comparison.
<p float="left">
  <img src="results/cvae/iter_965.png" width="400" />
  <img src="results/cvae/original.png" width="400" /> 
</p>

### Conditional generation
Using ```synthetic.py```, we can sample from the prior distribution of the CVAE, concatenate the vector with our desired ecnoding of the smile degree and let
the CVAE decode this sampled noise into a synthetic face of the desired smile degree. The range of smile-degree encodings in the training set is [-1,+1], where
+1 is most smiley, -1 is most non-smiley. Below side to side  64 synthetic images for encodings -0.5, +0.5 are shown produced with this method.

<p float="left">
  <img src="results/encode-0.5.png" width="400" />
  <img src="results/encode0.5.png" width="400" /> 
</p>

## Forcing smiles
With the trained model, one can use the pictures from the training set and instead of feeding in the  smile-degree encode of the corresponding picture we can fix an encoding or shift it by a factor to force the image a smile/non smile. Below this is done for 32 picture of the training set, on the op the original 32 images are shown, below the reconstruction with their actual encoding, and then we shift the encoding by +0.5, +0.7, -0.5, -0.7 to change the smile degree in the original picture (zoom in to see in detail!). Finally the same diagram is now shown for a single picture.

<img src="results/forcing_smiles_all.png" width="800" />

<img src="results/forcing_smiles_one.png" width="500" />


## The Dataset
The images of the faces come from [UTKFace Dataset](https://susanqq.github.io/UTKFace/). However the images do not have any encoding of a continuous degree of "smiley-ness". This "smile-strength" degree is produced by creating a slideshow of the images and exposing them to three subjects (me and a couple friends), by registering wheather the face was classified as smiley or non-smiley we encourage the subjects to answer as fast as possible so as to rely on first impression and the reaction time is registered.

## Notes: Bias in the Dataset
Its interesting to see that the when generating synthetic images with encodings < 0 (non-happy) the faces look more male-like and when generating synthetic images with encodings > 0 (happy) they tend to be more female-like. This is more apparent at the extremes, see the Note below. The original dataset although doesnt contains a smile degree encode, it has information of the image encoded in the filename, namely "gender" and "smile" as boolean values. Using this information then I can go and see if there was a bias in the dataset. In the piechart below the distribution of gender, and smile are shown. From there we can see that that although there are equals amount of men and women in the dataset, there were more non-smiley men than smiley men, and the bias of the synthetic generation may come from this unbalance.

<img src="results/dataset_bias.PNG" width="500" />

## Notes: Extending the encoding of smile-degree over the range for synthetic faces
Altough the range of smile-strength in the training set is [-1,+1], when generating synthetic images we can ask the model to generate outside of the range. But notice that then the synthetic faces become much more homogeneus, more than 64 different people it looks like small variations of the same synthetic image. Below side to side  64 synthetic images for encodings -3 (super not happy), +3 (super happy) are shown produced with this method.

<p float="left">
  <img src="results/encode-3.png" width="200" />
  <img src="results/encode3.png" width="200" /> 
</p>


## References:
* Fagertun, J., Andersen, T., Hansen, T., & Paulsen, R. R. (2013). 3D gender recognition using cognitive modeling. In 2013 International Workshop on Biometrics and Forensics (IWBF) IEEE. https://doi.org/10.1109/IWBF.2013.6547324
* Kingma, Diederik & Welling, Max. (2013). Auto-Encoding Variational Bayes. ICLR. 
* Learning Structured Output Representation using Deep Conditional Generative Models, Kihyuk Sohn, Xinchen Yan, Honglak Lee
