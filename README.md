# ChoreoAI

![alt text](ChoreoAI_Zixuan_Wang/assets/image-1.png)

## Background
While the fields of technology and dance have historically not often intersected, recent years have seen the advent of AI-generated choreography using models trained on motion capture of a single dancer. This project will expand the state-of-the-art in this intersectional field by exploring duets featuring pairs of dancers, enabling choreography that features authentic interactions between humans & AI models.

## Task ideas
- Extract pose information from curated videos of dance duets
- Train a GNN and/or Transformer model to analyze this data and generate new duet interaction ideas

## Expected results
- Create a dataset of dynamic point-cloud data corresponding to extracted motion capture poses from videos of dance duets
- Train an AI model that can generate the movements of Dancer #2 conditioned on the inputs of Dancer #1 and/or invent new, physically-plausible duet phrases
- If time permits: Learn key relationships between parts of the body of each dancer that are integral to the dynamics of the duet
- We will collaborate with the original dancers to use the model outputs to inspire new performance material

## Projects
| Contributor | Approach | Repository Link | Blog Post |
|-----------------|-----------------|-----------------|-----------------|
| Luis Zerkowski  | Graph Neural Network  | [Repo Link](https://github.com/humanai-foundation/ChoreoAI/tree/main/ChoreoAI_Duet_ChorAIgraphy_Luis_Zerkowski) | [Blog Post](https://medium.com/@luisvz)
| Zixuan Wang  | Transformer and VAE  | [Repo Link](https://github.com/humanai-foundation/ChoreoAI/tree/main/ChoreoAI_Zixuan_Wang)  | [Blog Post](https://wang-zixuan.github.io/posts/2024/gsoc_2024)