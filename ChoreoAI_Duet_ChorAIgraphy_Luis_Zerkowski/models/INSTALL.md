# Duet ChorAIgraphy Set Up

## Environment Installation

Despite the complexities involved in setting up the pose extraction pipeline, the process of setting up the environment to run the models themselves is quite simple. You just need to create your Anaconda environment using the `environment.yml` file with the following command:

```
conda env create -f environment.yml -n ai_choreo
```

It is encouraged, however, to use Mamba for this task due to its superior ability to handle potential package incompatibilities and its significantly faster environment creation process. To do this, simply [install Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) and replace `conda` with `mamba` in the above command.

## Training Your Model

For training a model, two main options are available:

- The more interactive approach via the `NRI_variant_test.ipynb` notebook, where every pipeline parameter can be adjusted for a personalized experience. Simply open the notebook and start exploring.

- The more modular approach through the `train_nri.py` Python script, which utilizes other scripts in the folder. This version is ideal for longer training sessions or use in computing clusters. With this use case in mind, the `creating_jobs.py` file is provided as an example for setting up and submitting grid jobs on SLURM clusters, enabling the exploration of various hyperparameters.