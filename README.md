# ChildGAN: Face Aging and Rejuvenation to Find Missing Children

Contributors: Praveen Kumar Chandaliya and **Neeta Nain**

# MRCD Dataset Agreement Form
https://github.com/praveenkumarchandaliya/ChildGAN_Tamp1/blob/main/MRCD%20Dataset%20Agreement%20Form.pdf

The images are labeled “age_gender_sequenceID, " where age is the person's age, and gender is the person's id, i.e., 0 or 1. For boys and girls, 0 and 1 are used as gender id, respectively.

# Research project Grant and Institutions
Multi-Racial Child Dataset is developed at the Department of Computer Science and Engineering of the **Malaviya  National Institute of Technology Jaipur** as part of a research project under grant No. 4 (13)/2019-ITEA by the **Ministry of Electronics and Information Technology (MeitY), Government of India**. 



# Description of MRCD dataset:  
The Multi-Racial Child Dataset (MRCD) contains 64,965 face images spanning four racial groups: Asian, Black, White, and Indian.

Distribution across racial groups:

Asian: 17,211 face images

Black: 13,354 face images

White: 19,297 face images

Indian: 15,103 face images

The dataset is intended strictly for research and academic purposes.

Each image has a resolution of 128 × 128 pixels, with a DPI range of 72–96.

The Asian, Black, and White child image subsets (MRCD) are used to train the ChildGAN model, and include images collected through web crawling and publicly available sources.

Images are labeled using the format:
age_genderId_sequenceID

age: age of the child

genderId: gender label (0 = boy, 1 = girl)

sequenceID: unique image identifier

Additional details regarding dataset construction, preprocessing, and usage protocols are available in the associated publications.

# Directory structure: <br />

   (Asian, Black, White) root directory--> <br />
     &nbsp; &nbsp;  &nbsp;  00--->0-3 Years Boys <br />
     &nbsp; &nbsp;  &nbsp;   01--->0-3 Years Girls <br />
     &nbsp; &nbsp;  &nbsp; 02--->4-8 Years Boys <br />
     &nbsp; &nbsp;  &nbsp; 03--->4-8 Year Girls <br />
     &nbsp; &nbsp;  &nbsp; 04--->9-12 Years Boys <br />
     &nbsp; &nbsp;  &nbsp; 05--->9-12 Year Girls <br />
     &nbsp; &nbsp; &nbsp;  06--->13-16 Years Boys <br />
     &nbsp; &nbsp; &nbsp;  07--->13-16 Year Girls <br />
     &nbsp; &nbsp; &nbsp;  08--->17-20 Years Boys <br />
     &nbsp; &nbsp; &nbsp;  09--->17-20 Year Girls <br />

# Dataset link for download: <br/>
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
[Praveen Kumar Chandaliya](https://scholar.google.com/citations?user=cx-vENIAAAAJ&hl=en) and [Neeta Nain](https://scholar.google.com/citations?user=CWsTU7EAAAAJ&hl=en). "ChildGAN: Face aging and rejuvenation to find missing children". Journal of Pattern Recognition Elsevier, 2022 (https://www.researchgate.net/publication/360289072_ChildGAN_Face_Aging_and_Rejuvenation_to_Find_Missing_Children).
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



