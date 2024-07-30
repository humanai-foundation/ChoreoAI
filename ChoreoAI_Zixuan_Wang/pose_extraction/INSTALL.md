# AlphaPose Installation on Perlmutter

## Perlmutter Introduction
[Perlmutter](https://docs.nersc.gov/getting-started/) is the computing resources for National Energy Research Scientific Computing Center (NERSC). It is a HPE Cray EX supercomputer with over 1500 GPU-accelerated compute nodes. 

To install AlphaPose on Perlmutter, please make sure that you have available GPU resources. It's very hard to use AlphaPose with CPU because it uses CUDA operators. 

Then, please follow the instructions below. The package version might be tricky.

## NVCC

`nvcc` is already installed in the login node of Perlmutter. The CUDA driver version on Perlmutter is 12.2, which requires us to install PyTorch >= 2.x.

```bash
nvcc --version
# 12.2
```

## Package Installation

- Anaconda: conda is easy for us to manage the python environment.

  ```bash
  wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
  bash Anaconda3-2023.03-Linux-x86_64.sh
  
  source ~/.bashrc
  # check conda version
  conda --version
  ```

- Installation of torch & torchvision

  ```bash
  # important! we need python version >= 3.10 to make everything work
  conda create -n alphapose python=3.10
  conda activate alphapose
  
  # install pytorch with wheel
  wget https://download.pytorch.org/whl/cu121/torch-2.3.0%2Bcu121-cp310-cp310-linux_x86_64.whl
  pip install torch-2.3.0%2Bcu121-cp310-cp310-linux_x86_64.whl
  
  # install torchvision with wheel
  wget https://download.pytorch.org/whl/cu121/torchvision-0.18.0%2Bcu121-cp310-cp310-linux_x86_64.whl
  pip install torchvision-0.18.0%2Bcu121-cp310-cp310-linux_x86_64.whl
  ```

- Get Alphapose from GitHub

  ```bash
  git clone https://github.com/MVIG-SJTU/AlphaPose.git
  cd alphapose
  ```

  ```bash
  # install dependencies
  pip install cython
  pip install chumpy
  
  conda install -c fvcore -c iopath -c conda-forge fvcore iopath
  conda install -c bottler nvidiacub
  
  # pytorch3d compatible with torch
  pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.6+pt2.3.0cu121
  
  wget https://download.pytorch.org/whl/nightly/pytorch_triton-3.0.0%2B989adb9a29-cp310-cp310-linux_x86_64.whl
  pip install pytorch_triton-3.0.0%2B989adb9a29-cp310-cp310-linux_x86_64.whl
  ```

- Halpecocotool issue

  - When installing AlphaPose, you might encounter Halpecocotool issue. Follow this https://github.com/MVIG-SJTU/AlphaPose/issues/1195 to solve it before installing AlphaPose!

- Final Installation

  ```bash
  # installation of alphapose requires gcc > 9
  export CC=/usr/bin/gcc-12
  export CXX=/usr/bin/g++-12
  export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  
  python setup.py build develop
  ```

## Inference

  ```bash
  # video inference, put the video file under scripts folder
  python scripts/demo_3d_inference.py --cfg configs/smpl/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml --checkpoint pretrained_models/pretrained_w_cam.pth --video scripts/Dyads\ Rehearsal\ Leah.mov --outdir examples/res_3d
  ```