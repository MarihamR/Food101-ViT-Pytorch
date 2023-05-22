# Food101-ViT-Pytorch

##  Samples of [Food101- Dataset] (https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101)
![Dataset_Samples](food101.png)

## Model Framework
This is implemented by Vision Transformer (Vit_b_16): [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
and pretrained with [SWAG trunk weights](https://arxiv.org/abs/2201.08371).
These weights are composed of the original frozen SWAG trunk weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
The model and pretrained weights can be found through [TorchVision](https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.ViT_B_16_Weights)


## Accuracy Comparison for Vision Transformer with and without SWAG pre-trained weights

| Model         | Accuracy      |
| ------------- | ------------- |
| VIT_b16: ImageNet weights       | 68.9%         |
| VIT_b16: ImageNet_SWAG weights  | 93.6%         |
