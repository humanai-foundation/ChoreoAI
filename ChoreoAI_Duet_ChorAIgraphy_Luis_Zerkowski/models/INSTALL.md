# Duet ChorAIgraphy Environment Installation

Despite the complexities involved in setting up the pose extraction pipeline, the process of setting up the environment to run the models themselves is quite simple. You just need to create your Anaconda environment using the `environment.yml` file with the following command:

```
conda env create -f environment.yml -n ai_choreo
```

It is encouraged, however, to use Mamba for this task due to its superior ability to handle potential package incompatibilities and its significantly faster environment creation process. To do this, simply [install Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) and replace `conda` with `mamba` in the above command.