# AlphaPose Installation
The AlphaPose installation can be tricky and may fail at different stages. To help avoid problems, I documented the installation process using Docker containers. Here's a simple tutorial on how I successfully installed AlphaPose on a Debian-based Linux OS (Ubuntu 22.04) with an NVIDIA RTX 4070 8GB and CUDA 12.2, using a Docker container to ensure compatibility.

## Set Up

First, ensure your graphics card is accessible. Open the terminal and run the command `nvidia-smi`. If this command shows the status of your graphics card and CUDA version, you're ready to proceed. If not, refer to [online tutorials](https://www.cherryservers.com/blog/install-cuda-ubuntu) or the [official documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) to properly install CUDA and `nvcc`.

Next, install Docker. There are many [online tutorials](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04) to guide you through this. You also need to install the NVIDIA container toolkit to use your GPU in a Docker container. Follow the [official documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for this installation.

With CUDA and Docker set up, we start the installation process for AlphaPose. Open your terminal and let's start running some commands:

```
# Pulling PyTorch image from docker hub
docker pull pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Find the IMAGE ID
docker images

# And run your container allowing for GPU usage
docker run --gpus all -it <IMAGE ID>

# Updates the list of available packages
apt update

# Installing system dependencies
apt install libyaml-dev
apt install locales
export LANG=C.UTF-8

# Installing python dependencies with correct and compatible versions
pip install torchaudio==2.3.0+cu121 -f https://download.pytorch.org/whl/torchaudio/
pip install cython
pip install chumpy
pip install easydict munkres natsort opencv-python pyyaml scipy tensorboardx terminaltables timm==0.1.20 tqdm visdom jinja2 typeguard
```

Now we'll adjust the `setup.py` file a little bit to make sure we install `halpecocotools` with no compatibility problems. First run `pip install git+https://github.com/Ambrosiussen/HalpeCOCOAPI.git#subdirectory=PythonAPI`. Then install nano `apt install nano` and use it `nano setup.py` to update the function `get_install_requires()` by removing `halpecocotools` from the `install_requires` list, ending up with:

```
def get_install_requires():
    install_requires = [
        'six', 'terminaltables', 'scipy',
        'opencv-python', 'matplotlib', 'visdom',
        'tqdm', 'tensorboardx', 'easydict',
        'pyyaml',
        'torch>=1.1.0', 'torchvision>=0.3.0',
        'munkres', 'timm==0.1.20', 'natsort'
    ]
```

And now we proceed with the installations:

```
# Installing visualization dependencies
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub

# Installing 3D libraries
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.6+pt2.3.0cu121
pip install pytorch_triton==3.0.0 -f https://download.pytorch.org/whl/nightly/pytorch-triton/

# Fixing 'numpy' potential incompatibilites with 'cython'
pip install "numpy<1.24"

# Installing git and cloning repo
apt install git
git clone https://github.com/MVIG-SJTU/AlphaPose.git
cd AlphaPose

# Building the package
python setup.py build develop
```

At this point, everything should already be properly installed. All you have to do now is download some models and put them in your container with `docker cp` to finally use the system.

- [Download the object detection model](https://drive.google.com/file/d/1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC/view) and place it into `AlphaPose/detector/yolo/data/`

- [Download human model for tracking](https://drive.google.com/file/d/1myNKfr2cXqiHZVXaaG8ZAq_U2UpeOLfG/view) and place it into `AlphaPose/trackers/weights/`. Rename it to `osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth`

- For 3D pose extraction and mesh reconstruction, [download human model](https://huggingface.co/spaces/brjathu/HMR2.0/blob/e5201da358ccbc04f4a5c4450a302fcb9de571dd/data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl) and place it into `AlphaPose/model_files/`

- Download any pose extraction model you want to run from the [AlphaPose's model zoo](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md) and place them into `AlphaPose/pretrained_models/`. 

## Running Models

Once everything is set up, you just need to run your selected model. Here's an example of how to do it:

```
# Go to AlphaPose root
cd AlphaPose

# Run a model for 2D inference
python scripts/demo_inference.py --cfg configs/smpl/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml --checkpoint pretrained_models/pretrained_w_cam.pth --indir <PATH_TO_YOUR_IMAGES> --outdir <OUTPUT_PATH_OF_YOUR_CHOICE> --save_img

# Or run the model for 3D inference
python scripts/demo_3d_inference.py --cfg configs/smpl/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml --checkpoint pretrained_models/pretrained_w_cam.pth --video <PATH_TO_YOUR_VIDEO> --outdir <OUTPUT_PATH_OF_YOUR_CHOICE> --save_img
```

Finally if you run into multi-processing problems with inference, making your program get stuck, [this issue](https://github.com/MVIG-SJTU/AlphaPose/pull/1064) might help.