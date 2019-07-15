# GloRe
Implementation for: [Graph-Based Global Reasoning Networks](https://arxiv.org/abs/1811.12814) (CVPR19)

## Software
- Image recognition experiments are in [MXNet \@92053bd](https://github.com/cypw/mxnet/tree/92053bd3e71f687b5315b8412a6ac65eb0cc32d5)
- Video and segmentation experiments are in PyTorch (0.5.0a0+783f2c6)


## Train & Evaluate

Train kinetics (single node):
```
./run_local.sh
```

Train kinetics (multiple nodes):
```
# please setup ./Host before running
./run_dist.sh
```

Evaluate the trained model on kinetics:
```
cd test
# check $ROOT/test/*.txt for the testing log
python test-single-clip.py
```

Note:
- The code is adapted from [MFNet (ECCV18)](https://github.com/cypw/PyTorch-MFNet).
- ImageNet pretrained models \([R50](https://dl.fbaipublicfiles.com/glore/kinetics/pretrained/resnet50-lite.pth), [R101](https://dl.fbaipublicfiles.com/glore/kinetics/pretrained/resnet101-lite.pth)\) might be required. Please put it under `$ROOT/network/pretrained/`.
- For image classification and segmentation tasks, please refer the code below.


## Results

### Image Recognition (ImageNet-1k)

Model                 |     Method     |  Res3  |  Res4  |   Code & Model   |  Top-1
:---------------------|:--------------:|:------:|:------:|:----------------:|:-------:
ResNet50              |    Baseline    |        |        | [link](https://dl.fbaipublicfiles.com/glore/imagenet/resnet50.tar) | 76.2 %
ResNet50              |    w/ GloRe    |        |   +3   | [link](https://dl.fbaipublicfiles.com/glore/imagenet/resnet50_w-glore_0-3.tar) | 78.4 %
ResNet50              |    w/ GloRe    |   +2   |   +3   | [link](https://dl.fbaipublicfiles.com/glore/imagenet/resnet50_w-glore_2-3.tar) | 78.2 %
SE-ResNet50           |    Baseline    |        |        | [link](https://dl.fbaipublicfiles.com/glore/imagenet/resnet50_w-se.tar) | 77.2 %
SE-ResNet50           |    w/ GloRe    |        |   +3   | [link](https://dl.fbaipublicfiles.com/glore/imagenet/resnet50_w-se_w-glore_0-3.tar) | 78.7 %

Model                 |     Method     |  Res3  |  Res4  |   Code & Model   |  Top-1
:---------------------|:--------------:|:------:|:------:|:----------------:|:-------:
ResNet200             |    w/ GloRe    |        |   +3   | [link](https://dl.fbaipublicfiles.com/glore/imagenet/resnet200_w-glore_0-3.tar) | 79.4 %
ResNet200             |    w/ GloRe    |   +2   |   +3   | [link](https://dl.fbaipublicfiles.com/glore/imagenet/resnet200_w-glore_2-3.tar) | 79.7 %
ResNeXt101 (32x4d)    |    w/ GloRe    |   +2   |   +3   | [link](https://dl.fbaipublicfiles.com/glore/imagenet/resnext101-32x4d_w-glore_2-3.tar) | 79.8 %
DPN-98                |    w/ GloRe    |   +2   |   +3   | [link](https://dl.fbaipublicfiles.com/glore/imagenet/dpn98_w-glore_2-3.tar) | 80.2 %
DPN-131               |    w/ GloRe    |   +2   |   +3   | [link](https://dl.fbaipublicfiles.com/glore/imagenet/dpn131_w-glore_2-3.tar) | 80.3 %

\* We use pre-activation[1] and strided convolution[2] for all networks for simplicity and consistency.


### Video Recognition (Kinetics-400)

Model                 | input frames | stride | Res3 | Res4 |    Model    | Clip Top-1
:---------------------|:------------:|:------:|:----:|:----:|:-----------:|:----------:
Res50 \(3D\) + Ours  |      8       |    8   |  +2  |  +3  | [link](https://dl.fbaipublicfiles.com/glore/kinetics/resnet50-lite_3d_8x8_w-glore_2-3_ep-0000.pth)  |   68.0 \% 
Res101 \(3D\) + Ours |      8       |    8   |  +2  |  +3  | [link](https://dl.fbaipublicfiles.com/glore/kinetics/resnet101-lite_3d_8x8_w-glore_2-3_ep-0000.pth) |   69.2 \% 

\* ImageNet-1k pretrained models: R50\([link](https://dl.fbaipublicfiles.com/glore/kinetics/pretrained/resnet50-lite.pth)\), R101\([link](https://dl.fbaipublicfiles.com/glore/kinetics/pretrained/resnet101-lite.pth)\).


### Semantic Segmentation (Cityscapes)

Method             | Backbone  | Code & Model |  IoU cla. | iIoU cla. | IoU cat. | iIoU cat.
:------------------|:---------:|:------------:|:---------:|:---------:|:--------:|:---------:
FCN + 1 GloRe unit | ResNet50  | [link](https://dl.fbaipublicfiles.com/glore/cityscapes/resnet50_w-glore.tar) | 79.5% | 60.3% | 91.3% | 81.5%
FCN + 1 GloRe unit | ResNet101 | [link](https://dl.fbaipublicfiles.com/glore/cityscapes/resnet101_w-glore.tar) | 80.9% | 62.2% | 91.5% | 82.1%

\* All networks are evaluated on Cityscapes test set by the testing server without using extra “coarse” training set.


## Other Resources

ImageNet-1k Training/Validation List:
- Download link: [GoogleDrive](https://goo.gl/Ne42bM)

ImageNet-1k category name mapping table:
- Download link: [GoogleDrive](https://goo.gl/YTAED5)

Kinetics Dataset:
- Downloader: [GitHub](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)

Cityscapes Dataset:
- Download link: [GoogleDrive](https://www.cityscapes-dataset.com)


## FAQ

### Where can I find the code for image classification and segmentation?
- The code is packed with the model within the same `*.tar` file.

### Do I need to convert the raw videos to specific format?
- The `dataiter' supports reading from raw videos.

### How can I make the training faster?
- Remove HLS augmentation (won't make much difference); Try to convert the raw videos to lower resolution to reduce the decoding cost (We use <=288p for all experiment). 

For example:
```
# convet to sort_edge_length <= 288
ffmpeg -y -i ${SRC_VID} -c:v mpeg4 -filter:v "scale=min(iw\,(288*iw)/min(iw\,ih)):-1" -b:v 640k -an ${DST_VID}
# or, convet to sort_edge_length <= 256
ffmpeg -y -i ${SRC_VID} -c:v mpeg4 -filter:v "scale=min(iw\,(256*iw)/min(iw\,ih)):-1" -b:v 512k -an ${DST_VID}
# or, convet to sort_edge_length <= 160
ffmpeg -y -i ${SRC_VID} -c:v mpeg4 -filter:v "scale=min(iw\,(160*iw)/min(iw\,ih)):-1" -b:v 240k -an ${DST_VID}
```

## Reference
```
[1] He, Kaiming, et al. "Identity mappings in deep residual networks."
[2] https://github.com/facebook/fb.resnet.torch
```

## Citation
```
@inproceedings{chen2019graph,
  title={Graph-based global reasoning networks},
  author={Chen, Yunpeng and Rohrbach, Marcus and Yan, Zhicheng and Shuicheng, Yan and Feng, Jiashi and Kalantidis, Yannis},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={433--442},
  year={2019}
}
```

## License
The code and the models are MIT licensed, as found in the LICENSE file.
