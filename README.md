# SIFT Reconstruction

An official source code 
for paper "Deep Reverse Attack on SIFT Features With a Coarse-to-Fine GAN Model"

## Table of Contents

- [Background](#background)
- [Dependency](#dependency)
- [Demo](#demo)
- [Citation](#citation)


## Background
In this work, we conduct reverse attacks for evaluating the privacy leakage risk of Scale Invariant Feature Transform (SIFT).

 We propose a two-stage deep generation model called Coarse-to-Fine Generative Adversarial Network (CFGAN) to conduct reverse attacks for evaluating the privacy leakage risk of SIFT features. Specifically, the proposed model consists of two sub-networks, namely coarse net and fine net. The coarse net is developed to restore coarse images using SIFT features, while the fine net is responsible for refining the coarse images to obtain better reconstruction results. 

<p align='center'>  
  <img src='https://github.com/HITLiXincodes/CFGAN/blob/main/CFGAN/CFGAN/images/whole.png' width='870'/>
</p>
<p align='center'>  
  <em>Framework of CFGAN model.</em>
</p>

To effectively leverage the information contained in SIFT features, an efficient fusion strategy based on the AdaIN operation is designed in the fine net. Additionally, we introduce a new loss function called sift loss to enhance the color fidelity of reconstructed images.


## Dependency
- torch 1.13.0
- opencv 4.7.0
- python 3.9.12

To train or test the CFGAN model, please download datasets from their official websites, and put them under the `./dataset/` directory.
The pre-trained models should be put under the `./weights/` directory.

**Note: The pretrained weights can be downloaded from:
[Baidu Yun (Code: )]()**

## Demo

To train or test the CFGAN model:
```bash
python CFGAN_main.py {train,test}
```

For example to test the CFGAN model:
```bash
python CFGAN_main.py test
```
Then the model will reconstruct the images in the `./dataset/test/` and save the results in the `./res/CFGAN/` directory.
For more details please refer to the paper.

## Citation

If you use this code for your research, please cite our paper
