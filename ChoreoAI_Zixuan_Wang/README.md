## Introduction
This is the repo for [Google Summer of Code 2024 - AI-Generated Choreography - from Solos to Duets](https://humanai.foundation/gsoc/2024/proposal_ChoreoAI1.html). 

Blog post for summarizing my work: [blog post](https://wang-zixuan.github.io/posts/2024/gsoc_2024).

Contributor page of GSoC: [Zixuan Wang](https://summerofcode.withgoogle.com/programs/2024/projects/PEvVr15z). Evaluation test repo: [repo link](https://github.com/wang-zixuan/AI_Enabled_Choreography).

I would like to thank my advisors [Mariel Pettee](https://marielpettee.com/) and [Ilya Vidrin](https://www.ilyavidrin.com/) for their guidance, and my partner [Luis Zerkowski](https://github.com/Luizerko) for his help.

## Goals
### Problem Addressed
This project aims at developing AI-generated choreography for duets. It outlines a novel approach leveraging Transformer and Variational Autoencoder to analyze and generate dance movements, focusing on creating authentic dance material that reflect the intricate dynamics between dancers.

### Solution Approach 
The project will use Transformer and Variational Autoencoder to process motion capture data. 

The comprehensive solution plan involves:
- Understanding key relationships between parts of the body of each dancer through a dedicated VAE containing self-attention layers and LSTM layers
- Desining a Duet VAE to learn key interaction between two dancers
- Enhancing the overall coherence of the generated choreography by implementing regression loss and velocity loss

### Deliverables
- Create a dataset of point-cloud data corresponding to extracted motion capture poses from videos of dance duets
- Utilize Transformer model with encoder-decoder structure and various attention mechanisms to generate the movements of Dancer #2 conditioned on the inputs of Dancer #1 and/or invent new, physically-plausible duet phrases
- Implement a Spatial/Temporal Joint Relation Module to learn key relationships between parts of the body of each dancer that are integral to the dynamics of the duet
- If time permits: incorporate Music Embedding Module to enrich the model with multi-modal inputs
- Collaborate with the original dancers to use the model outputs to inspire new performance material

## Repository Hierarchy
```
.
├── data
│   └── dataset.py
├── dataset
│   ├── pose_extraction_dyads_rehearsal_leah.npy
│   ├── pose_extraction_hannah_cassie.npy
│   ├── pose_extraction_ilya_hannah_dyads.npy
│   └── pose_extraction_img_9085.npy
├── loss
│   └── reconstruction_loss.py
├── model
│   ├── model_pipeline.py
│   └── transformer.py
├── pose_extraction
│   ├── imgs
│   ├── INSTALL.md
│   ├── preprocessing.ipynb
│   └── README.md
├── README.md
├── result
│   └── best_model.pth
├── test.py
├── train.py
├── utils
│   ├── logger.py
│   └── misc.py
└── visualization.ipynb
```

## Solution
### Data preparation
Please refer to [this document](pose_extraction/README.md) for more details.

### Model Structure
<div align="center">
  <img src="assets/network.png" width="50%" />
</div>

The model includes 3 VAEs, separately for dancer 1, dancer 2, and duet. Each VAE contains encoder and decoder, and encoder contains self-attention layer and LSTM layer. VAE for duet will receive original data of dancer 1 and dancer 2, and simply calculate the distance between two dancers to measure proximity. Then, it will generate two sequences representing interrelation for each dancer.