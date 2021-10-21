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


## The Dataset
The images of the faces come from [UTKFace Dataset](https://susanqq.github.io/UTKFace/). However the images do not have any encoding of a continuous degree of "smiley-ness". This "smile-strength" degree is produced by creating a slideshow of the images and exposing them to three subjects (me and a couple friends), by registering wheather the face was classified as smiley or non-smiley we encourage the subjects to answer as fast as possible so as to rely on first impression and the reaction time is registered.


## Notes: Extending the encoding of smile-degree over the range for synthetic faces
Altough the range of smile-strength in the training set is [-1,+1], when generating synthetic images we can ask the model to generate outside of the range. But notice that then the synthetic faces become much more homogeneus, more than 64 different people it looks like small variations of the same synthetic image. Below side to side  64 synthetic images for encodings -3 (super not happy), +3 (super happy) are shown produced with this method.

<p float="left">
  <img src="results/encode-3.png" width="300" />
  <img src="results/encode3.png" width="300" /> 
</p>


## References:
* Fagertun, J., Andersen, T., Hansen, T., & Paulsen, R. R. (2013). 3D gender recognition using cognitive modeling. In 2013 International Workshop on Biometrics and Forensics (IWBF) IEEE. https://doi.org/10.1109/IWBF.2013.6547324
* Kingma, Diederik & Welling, Max. (2013). Auto-Encoding Variational Bayes. ICLR. 
* Learning Structured Output Representation using Deep Conditional Generative Models, Kihyuk Sohn, Xinchen Yan, Honglak Lee
