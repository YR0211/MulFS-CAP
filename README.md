#  MulFS-CAP

### MulFS-CAP: Multimodal Fusion-supervised Cross-modality Alignment Perception for Unregistered Infrared-visible Image Fusion
By Huafeng Li; Zengyi Yang; Yafei Zhang; Wei Jia; Zhengtao Yu; Yu Liu*

Our paper is available online! [[IEEE](https://ieeexplore.ieee.org/document/10856402)]

<div align=center>
<img src="https://github.com/YR0211/MulFS-CAP/blob/main/overview.png" width="90%">
</div>

## Requirements

 - [ ] torch  1.12.1

 - [ ] torchvision 0.13.1

 - [ ] opencv  4.6.0.66

 - [ ] kornia  0.5.11

 - [ ] numpy  1.21.5
 

## To Test
1.If you want to test input source images with a fixed resolution of 256x256, you can run following commands:
    
    python test.py
    
2.If you want to test input source images of arbitrary resolution, you can run following commands:

    python test_arbitrary_resolution.py

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
  title={MulFS-CAP: Multimodal Fusion-Supervised Cross-Modality Alignment Perception for Unregistered Infrared-Visible Image Fusion}, 
  year={2025},
  volume={47},
  number={5},
  pages={3673-3690},
}
```
