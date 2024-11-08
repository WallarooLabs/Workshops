## Models for Wallaroo Computer Vision Tutorials

In order for the wallaroo tutorial notebooks to run properly, the videos directory must contain these models in the models directory.

To download the Wallaroo Computer Vision models:

1. Install the Google GCloud terminal application.
1. Use the following cmd in a terminal, preferably in the `./models` directory:

    ```bash
    gcloud storage "cp gs://wallaroo-model-zoo/open-source/computer-vision/models/*" .
    ```

The other source is the Wallaroo Workspaces GitHub repository Release site:

[Wallaroo Workshops Releases](https://github.com/WallarooLabs/Workshops/releases/tag/1.0-initial-release)

Download the `Computer-Vision-Retail.zip` file - this includes the sample models.

### Directory contents

* coco_classes.pickle - contain the 80 COCO classifications used by resnet50 and mobilenet object detectors.  
* frcnn-resent.pt - PyTorch resnet50 model
* frcnn-resnet.pt.onnx - PyTorch resnet50 model converted to onnx
* mobilenet.pt - PyTorch mobilenet model
* mobilenet.pt.onnx - PyTorch mobilenet model converted to onnx
