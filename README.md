#  MulFS-CAP

### MulFS-CAP: Multimodal Fusion-supervised Cross-modality Alignment Perception for Unregistered Infrared-visible Image Fusion
By Huafeng Li; Zengyi Yang; Yafei Zhang; Wei Jia; Zhengtao Yu; Yu Liu*

[paper](https://ieeexplore.ieee.org/document/10856402)

## Requirements

 - [ ] torch  1.12.1

 - [ ] torchvision 0.13.1

 - [ ] opencv  4.6.0.66

 - [ ] kornia  0.5.11

 - [ ] numpy  1.21.5
 

## To Test
    python test.py

## To Train
    python train.py

## Pretrained Model
*   The pretrained model on the RoadScene dataset is as follows: [RoadScene](https://drive.google.com/drive/folders/14RwjzYiTPThZSSruAe_8XY9SAajQrlrF?usp=sharing) (Google Link)
*   If you intend to evaluate the deformed images you constructed, retraining the model is recommended.

## Citation
```
@ARTICLE{MulFS-CAP,
  author={Li, Huafeng and Yang, Zengyi and Zhang, Yafei and Jia, Wei and Yu, Zhengtao and Liu, Yu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={MulFS-CAP: Multimodal Fusion-supervised Cross-modality Alignment Perception for Unregistered Infrared-visible Image Fusion}, 
  year={2025},
  pages={1-18}
}
```
