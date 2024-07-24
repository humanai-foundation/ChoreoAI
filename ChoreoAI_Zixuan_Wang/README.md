## Introduction
This is the repo for [Google Summer of Code 2024 - AI-Generated Choreography - from Solos to Duets](https://humanai.foundation/gsoc/2024/proposal_ChoreoAI1.html). 

Contributor page of GSoC: [Zixuan Wang](https://summerofcode.withgoogle.com/programs/2024/projects/PEvVr15z). Evaluation test repo: [repo link](https://github.com/wang-zixuan/AI_Enabled_Choreography).

I would like to thank my advisors [Mariel Pettee](https://marielpettee.com/) and [Ilya Vidrin](https://www.ilyavidrin.com/) for their guidance, and my partner [Luis Zerkowski](https://github.com/Luizerko) for his help.

## Goals
### Problem Addressed
This project aims at developing AI-generated choreography for duets. It outlines a novel approach leveraging Transformer to analyze and generate dance movements, focusing on creating authentic dance material that reflect the intricate dynamics between dancers.

### Solution Approach 
The project will use Transformer to process motion capture data. 

The comprehensive solution plan involves:
- Understanding key relationships between parts of the body of each dancer through a dedicated Spatial/Temporal Joint Relation Module
- Utilizing self-attention and cross-attention mechanisms to learn spatial-temporal relationships of dancers' movements
- Enhancing the overall coherence of the generated choreography by implementing regression loss and style similarity loss 

### Deliverables
- Create a dataset of point-cloud data corresponding to extracted motion capture poses from videos of dance duets
- Utilize Transformer model with encoder-decoder structure and various attention mechanisms to generate the movements of Dancer #2 conditioned on the inputs of Dancer #1 and/or invent new, physically-plausible duet phrases
- Implement a Spatial/Temporal Joint Relation Module to learn key relationships between parts of the body of each dancer that are integral to the dynamics of the duet
- If time permits: incorporate Music Embedding Module to enrich the model with multi-modal inputs
- Collaborate with the original dancers to use the model outputs to inspire new performance material

## Repository Hierarchy
### pose_extraction

### model

### dataset