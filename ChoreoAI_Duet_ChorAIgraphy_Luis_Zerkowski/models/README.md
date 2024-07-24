# Models

After extensive data processing, attention was turned to developing the models themselves. The use of AI in dance allows for a wide range of creative possibilities. Many different tasks can be explored, but two stand out in this project: dance interpretability and the generation of new phrases.

The first task involves studying the hidden movements of a duet, extracting information that is not visually evident. For example, it might not be immediately clear that the movement of one dancer's right foot is directly connected to the movement of the other dancer's hip or left hand. The objective is to model and tackle a **Graph Structure Learning** problem, uncovering the connections (or different types of connections) between the dancers' joints. More specific technical details are described in a dedicated section below.

The second task is a natural continuation of the first. Using the connections learned from the first model or maybe even a graph structure defined by a user, the goal is to **Generate New Dance Sequences** guided by these connections. More clearly, this pipeline aims to create new movements that follow a suggested line. The technical details of this model are also provided in its respective section below, although it remains in the conceptual stage at this point in the project.

## Set Up

Refer to the [installation document](https://github.com/Luizerko/ai_choreo/blob/master/models/INSTALL.md) to set up your own version of the project.

## Loading Data

The next step in the project involves loading the dance data for AI modeling. First, the preprocessed data is read in an interleaved manner to separately extract data from both dancers. Adjacencies are then created by initializing a default skeleton with 29 joints for each dancer and connecting every joint of one dancer to all joints of the other.

<div align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/models/assets/skeleton_and_connections.png", width="450">
</div>
<div align='center'>
    <span>Plot of the two dancers with only their skeletons (left) and fully connected to their partner (right).</span>
    <br><br>
</div>

The idea behind mapping the skeletons of both dancers in this way is to ensure the model focuses on the connections between them, rather than on the connections within each individual dancer. It's natural that much of a dancer's joint movement could be more easily predicted by inspecting their other joints. However, the goal here is to focus on the influences between the dancers, identifying which joints of one dancer influence the other and vice versa. It is also worth noting that, to simplify the initial modeling, this graph is undirected. This approach will be adjusted in the future to evaluate the direction of the influences between each joint of both dancers.

These connections are what the graph structure learning model will classify, initially as existing or non-existing edges for simplicity, but later categorically by the degree of influence in the movement. The data is then prepared for model training by creating batches with PyTorch tensors. The tensors are structured with dimensions representing the total number of sequences, the sequence length, the number of joints from both dancers, and 3D coordinates. Finally, a training-validation split is created to allow for proper model hyperparameter tuning.

## Neural Relational Inference Variant

As the title suggests, this model is a variant of the [Neural Relational Inference (NRI)](https://arxiv.org/abs/1802.04687) model, which itself is an extension of the traditional [Variational Autoencoder (VAE)](https://arxiv.org/abs/1312.6114). The primary objective of the original model is to study particles moving together in a system without prior knowledge of their underlying relationships. By analyzing their movement (position and velocity), the model estimates a graph structure that connects these particles, aiming to reveal which particles exert force on the others.

In the context of this project, the particles are represented by the joints of dancers. While the physical connections between joints within a dancer's body are known, this information alone is insufficient to understand the partnering relationships between two dancers.

Since a target graph structure correctly identifying which joints are virtually connected during a dance performance is unavailable, and considering that this graph can change over time even within a performance, self-supervising techniques are employed - one of the reasons for choosing the VAE framework.

The model consists of an encoder and a decoder, both playing around with transforming node representations into edge representations and vice versa. This approach emphasizes the dynamics of movements rather than fixed node embeddings. Not only that, but the encoder specifically outputs edges, sampling these from the generated latent space, making it essential to switch between representations.

<div align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/models/assets/final_arch.png", width="550">
</div>
<div align="center">
    <span>Image adapted from the <a href='https://arxiv.org/abs/1802.04687'>original NRI paper</a> showing a schematic overview of the final model architecture, including the GCN nodes and the sequence-to-sequence adapatation.</span>
    <br><br>
</div>

This project's implementation, even though very similar to the NRI MLP-Encoder MLP-Decoder model, includes a few important modifications:

- Graph Convolutional Network (GCN): Some MLP layers are replaced with GCN layers to leverage the graph structure, improving the model's ability to capture relationships between joints. This change focuses on a subset of edges connecting both dancers rather than studying all particle relationships as in the original implementation. Additionally, GCNs provide local feature aggregation and parameter sharing, important inductive biases for this context, resulting in enhanced generalization in a scenario with dynamic (and unknown) graph structures.

- Predicting Sequences: Since the data only includes noisy 3D position of points (and not their velocity), the Markovian property explored by NRI for reconstructions does not hold. Therefore, to predict movement, the model reconstructs entire (small) sequences.

- Use of Modern Library: PyTorch Geometric is utilized for its advanced features and ease of use.

By incorporating these modifications, the model maintains the core principles of the original NRI model while enhancing its ability to generalize and adapt to the dynamic nature of dance performances.

## Temporal Model
