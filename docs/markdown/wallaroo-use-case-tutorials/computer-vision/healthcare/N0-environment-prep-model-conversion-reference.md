This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-cookbooks/computer-vision).

## Step 00: Introduction and Setup

This tutorial demonstrates how to use the Wallaroo to detect objects in images through the following models:

* **rnn mobilenet**: A single stage object detector that performs fast inferences.  Mobilenet is typically good at identifying objects at a distance.
* **resnet50**:  A dual stage object detector with slower inferencing but but is able to detect objects that are closer to each other.

This tutorial series will demonstrate the following:

* How to deploy a Wallaroo pipeline with trained rnn mobilenet model and perform sample inferences to detect objects in pictures, then display those objects.
* How to deploy a Wallaroo pipeline with a trained resnet50 model and perform sample inferences to detect objects in pictures, then display those objects.
* Use the Wallaroo feature shadow deploy to have both models perform inferences, then select the inference result with the higher confidence and show the objects detected.

This tutorial assumes that users have installed the [Wallaroo SDK](https://pypi.org/project/wallaroo/) or are running these tutorials from within their Wallaroo instance's JupyterHub service.

This demonstration should be run within a Wallaroo JupyterHub instance for best results.

## Prerequisites

The included OpenCV class is included in this demonstration as `CVDemoUtils.py`, and requires the following dependencies:

* ffmpeg
* libsm
* libxext

### Internal JupyterHub Service

To install these dependencies in the Wallaroo JupyterHub service, use the following commands from a terminal shell via the following procedure:

1. Launch the JupyterHub Service within the Wallaroo install.
1. Select **File->New->Terminal**.
1. Enter the following:

    ```bash
    sudo apt-get update
    ```

    ```bash
    sudo apt-get install ffmpeg libsm6 libxext6  -y
    ```

### External SDK Users

For users using the Wallaroo SDK to connect with a remote Wallaroo instance, the following commands will install the required dependancies:

For Linux users, this can be installed with:

```bash
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y
```

MacOS users can prepare their environments using a package manager such as [Brew](https://brew.sh/) with the following:

```bash
brew install ffmpeg libsm libxext
```

### Libraries and Dependencies

1. This repository may use large file sizes for the models.  If necessary, install [Git Large File Storage (LFS)](https://git-lfs.com) or use the [Wallaroo Tutorials Releases](https://github.com/WallarooLabs/Wallaroo_Tutorials/releases) to download a .zip file of the most recent computer vision tutorial that includes the models.
1. Import the following Python libraries into your environment:
    1. [torch](https://pypi.org/project/torch/)
    1. [wallaroo](https://pypi.org/project/wallaroo/)
    1. [torchvision](https://pypi.org/project/torchvision/)
    1. [opencv-python](https://pypi.org/project/opencv-python/)
    1. [onnx](https://pypi.org/project/onnx/)
    1. [onnxruntime](https://pypi.org/project/onnxruntime/)
    1. [imutils](https://pypi.org/project/imutils/)
    1. [pytz](https://pypi.org/project/pytz/)
    1. [ipywidgets](https://pypi.org/project/ipywidgets/)

These can be installed by running the command below in the Wallaroo JupyterHub service.  Note the use of `pip install torch --no-cache-dir` for low memory environments.

```python
!pip install torchvision==0.15.2
!pip install torch==2.0.1 --no-cache-dir
!pip install opencv-python==4.7.0.72
!pip install onnx==1.12.0
!pip install onnxruntime==1.15.0
!pip install imutils==0.5.4
!pip install pytz
!pip install ipywidgets==8.0.6
!pip install patchify==0.2.3
!pip install tifffile==2023.4.12
!pip install piexif==1.1.3
```

The rest of the tutorials will rely on these libraries and applications, so finish their installation before running the tutorials in this series.

## Models for Wallaroo Computer Vision Tutorials

In order for the wallaroo tutorial notebooks to run properly, the videos directory must contain these models in the models directory.

To download the Wallaroo Computer Vision models, use the following link:

https://storage.googleapis.com/wallaroo-public-data/cv-demo-models/cv-retail-models.zip

Unzip the contents into the directory `models`.

### Directory contents

* coco_classes.pickle - contain the 80 COCO classifications used by resnet50 and mobilenet object detectors.  
* frcnn-resent.pt - PyTorch resnet50 model
* frcnn-resnet.pt.onnx - PyTorch resnet50 model converted to onnx
* mobilenet.pt - PyTorch mobilenet model
* mobilenet.pt.onnx - PyTorch mobilenet model converted to onnx

