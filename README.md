# TF2_U2Model
This is the U2 model for segmentation implemented in Tensorflow 2.3



## Reference from 
- U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection
  https://arxiv.org/pdf/2005.09007.pdf
- pytorch implemtation (We revised the code to tensorflow version)
  https://github.com/NathanUA/U-2-Net





![pterygium](https://github.com/soxHenry433/TF2_U2Model/blob/master/Test/1D0734605CB1FF86A792C14BB6A794616FA37246-HR-20181122_0.png "Predicted images")

|:---------:|:---------:|
|**Raw Image**|**Predicted**| 
|**Raw Mask**|**Predicted**|



| 		 		   | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 |
|:----------------:|:------:|:------:|:------:|:------:|:------:|:------:|
| **ResNet50**     |  0.89  |  0.89  |  0.89  |  0.89  |  0.89  |  0.89  |
| **ResNet101**    |  0.89  |  0.89  |  0.89  |  0.89  |  0.89  |  0.89  |
| **ResNet152**    |  0.89  |  0.89  |  0.89  |  0.89  |  0.89  |  0.89  |
| **DenseNet121**  |  0.89  |  0.89  |  0.89  |  0.89  |  0.89  |  0.89  |
| **DenseNet169**  |  0.89  |  0.89  |  0.89  |  0.89  |  0.89  |  0.89  |
| **Inception_V3** |  0.89  |  0.89  |  0.89  |  0.89  |  0.89  |  0.89  |

        
