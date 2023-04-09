# HumanSegmentation
The project involves training a U2-Net Lite model for human segmentation using the P3M-10k dataset. Human segmentation refers to the process of accurately separating the human body from the background in an image or video. 

## Environment Setup
* python >= 3.9 (Anaconda)
* torch
* torchvision
* opencv
* numpy
* Pillow

## Prepare Training Dataset
Used the **P3M-10K** dataset for training **U2Net Lite** model. The dataset contains ~10,000 images of people in various poses and environments. To augment the dataset, I applied horizontal flipping to the images, resulting in a total of ~20,000 training images. The dataset was split into training and validation sets with a ratio of 80:20.


## Acknowledgements
* U2net model is from original [u2net repo](https://github.com/xuebinqin/U-2-Net). Thanks to Xuebin Qin for amazing repo.
* Thanks to the creators of the [P3M-10K](https://github.com/JizhiziLi/P3M) dataset for providing a valuable resource.