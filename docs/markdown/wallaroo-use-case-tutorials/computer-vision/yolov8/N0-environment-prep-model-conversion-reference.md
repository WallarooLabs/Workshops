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
    sudo apt-get install ffmpeg libsm6 libxext6 libgl1 -y
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
brew install ffmpeg libsm libxext mesa-glu
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
    1. [ultralytics](https://pypi.org/project/ultralytics/)

These can be installed by running the command below in the Wallaroo JupyterHub service.  Note the use of `pip install torch --no-cache-dir` for low memory environments.

```python
!pip install torchvision
!pip install torch --no-cache-dir
!pip install opencv-python
!pip install onnx
!pip install onnxruntime
!pip install imutils
!pip install pytz
!pip install ipywidgets
!pip install ultralytics
```

The rest of the tutorials will rely on these libraries and applications, so finish their installation before running the tutorials in this series.
