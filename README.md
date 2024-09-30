# ChildGAN: Face Aging and Rejuvenation to Find Missing Children
By Praveen Kumar Chandaliya and Neeta Nain. 

# MRCD Dataset Agreement Form
https://github.com/praveenkumarchandaliya/ChildGAN_Tamp1/blob/main/MRCD%20Dataset%20Agreement%20Form.pdf

The images are labeled “age_gender_sequenceID, " where age is the person's age, and gender is the person's id, i.e., 0 or 1. For boys and girls, 0 and 1 are used as gender id, respectively.


# Link For Download the MRCD dataset:  
Asian, Black, and White children dataset image (MRCD) to train the ChildGAN model, including web crawl and publicly collected images.
The images are labeled in the format "age_genderId_sequenceID", where age is the age of the children, and genderId is the children's id, i.e., 0 or 1. For boys and girls, 0 and 1 are used as gender id, respectively.
Asian, Black, and White children dataset image (MRCD) to train the ChildGAN model, including web crawl and publicly collected images.

# Directory structure:
root directory-->
              00--->0-3 Years Boys
              01--->0-3 Years Girls
              02--->4-8 Years Boys
              03--->4-8 Year Girls
              04--->9-12 Years Boys
              05--->9-12 Year Girls
              06--->13-16 Years Boys
              07--->13-16 Year Girls
              08--->17-20 Years Boys
              09--->17-20 Year Girls



<a href="https://drive.google.com/file/d/1_jOclJy3AFbSHzKsuIh7QD-UOsb5p2RT/view?usp=drive_link">MRCD Dataset<a>


### Introduction

This repo is the official Pytorch implementation for our paper ChildGAN: Face Aging and Rejuvenation to Find Missing Children.

<div align="center">
<img align="center" src="images/ChildGAN.png" width="600" alt="ChildGAN Framework">
</div>
<div align="center">
Model Architecture.
</div>
<br/>

### Requirement

- Python 2.7 or higher
- Pytorch 

### Training and Testing ChildGAN

1. Train Model: `ChildGANTrain.py` file.

2. Test  Model: `ChildGANTest.py` file.

### Generalization Result

<div align="center">
<img align="center" src="images/SkinColorFinal.png" alt="Generalization">
</div>
<div align="center">
Age progressed faces on four race (a) Asian, (b) Black, (c) White and (d) Indian.
</div>
<br/>

## MRCD Dataset
MRCD Dataset present in CRFW directory: https://github.com/praveenkumarchandaliya/ChildGAN_Tamp1/tree/main/CRFW
## Citation
[Praveen Kumar Chandaliya](https://github.com/praveenkumarchandaliya/ChildGAN_Tamp1/) and [Neeta Nain](https://github.com/praveenkumarchandaliya/ChildGAN_Tamp1/). "ChildGAN: Face aging and rejuvenation to find missing children". Journal of Pattern Recognition Elsevier, 2022 (https://www.researchgate.net/publication/360289072_ChildGAN_Face_Aging_and_Rejuvenation_to_Find_Missing_Children).
```
@inproceedings{PraveenICD2022,
  title={ChildGAN: Face aging and rejuvenation to find missing children},
  author={Praveen Kumar Chandaliya, Neeta Nain},
  booktitle={Pattern Recognition},
  year={2022},
  volume = {129},
  pages = {108761}
}
@inproceedings{PraveenICD2022,
  title={Conditional Perceptual Adversarial Variational Autoencoder for Age Progression and Regression on Child Face},
  author={Praveen Kumar Chandaliya, Neeta Nain},
  booktitle={International Conference on Biometrics (ICB)},
  year={2019},
  pages = {1-8}
}

@inproceedings{PraveenSAMSP2021,
  title={Child Face Age Progression and Regression using Multi-Scale Patch GAN},
  author={Praveen Kumar Chandaliya, Neeta Nain},
  booktitle={International Joint Conference on Biometrics (IJCB)},
  year={2021},
  pages = {1-8}
}



@inproceedings{AWGAN2022,
  title={AWGAN: Face Age Progression and Regression using Attention},
  author={Praveen Kumar Chandaliya, Neeta Nain},
  booktitle={Neural Computing and Applications},
  year={2022},
  volume = {34},
  pages = {1-16}
}

 

```



