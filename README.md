# Fighting deepfakes by detecting GAN DCT anomalies

[Paper link](https://www.mdpi.com/2313-433X/7/8/128)
(This article belongs to the Special Issue Image and Video Forensics) - Journal of Imaging 2021 - MDPI


## Abstract

Synthetic multimedia contents created through AI technologies, such as Generative Adversarial Networks (GAN), applied to human faces can have serious social and political consequences. State-of-the-art algorithms employ deep neural networks to detect fake contents but, unfortunately, almost all approaches appear to be neither generalizable nor explainable. In this paper, a new fast detection method able to discriminate Deepfake images with high precision is proposed. By employing Discrete Cosine Transform (DCT), anomalous frequencies in real and Deepfake image datasets were analyzed. The β statistics inferred by the distribution of AC coefficients have been the key to recognize GAN-engine generated images. The proposed technique has been validated on pristine high quality images of faces synthesized by different GAN architectures. Experiments carried out show that the method is innovative, exceeds the state-of-the-art and also gives many insights in terms of explainability.

## Citing 

If using this code please cite

```
@article{giudice2021fighting,
   title={Fighting deepfakes by detecting GAN DCT anomalies },
   author={Giudice, Oliver and Guarnera, Luca and Battiato, Sebastiano},
   journal={Journal of Imaging},
   volume={7},
   number={8},
   pages={128}
   year={2021},
   url = {https://www.mdpi.com/2313-433X/7/8/128},
   DOI = {10.3390/jimaging7080128}
}
```

## How to install & use

```bash
conda env create -f environment.yml
conda activate dct2
python detect.py <filename>
```

You can use one of the provided files in sample_data