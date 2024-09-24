# Models

After extensive data processing, attention was turned to developing a model. The use of AI in dance allows for a wide range of creative possibilities. Many different tasks can be explored, but one stands out in this project: dance interpretability to help in the generation of new phrases.

The task involves studying the hidden movements of a duet, extracting information that is not visually evident. For example, it might not be immediately clear that the movement of one dancer's right foot is directly connected to the movement of the other dancer's hip or left hand. The objective thus is to model and tackle a **Graph Structure Learning** problem, uncovering the connections (or different types of connections) between the dancers' joints. More specific technical details are described in a dedicated section below.

<!-- The second task is a natural continuation of the first. Using the connections learned from the first model or maybe even a graph structure defined by a user, the goal is to **Generate New Dance Sequences** guided by these connections. More clearly, this pipeline aims to create new movements that follow a suggested line. The technical details of this model are also provided in its respective section below, although it remains in the conceptual stage at this point in the project. -->

Before getting into the setup and implementation details though, if the goal is simply to run one of the trained models and visualize the results, this can be done using the following [Colab Notebook](https://colab.research.google.com/drive/1KhX-Ppn9-BxAO4EX0BtfqPohdA09I6z5?usp=sharing).

## Set Up

Refer to the [installation document](https://github.com/Luizerko/ai_choreo/blob/master/models/INSTALL.md) to set up your own version of the project.

## Loading Data

The next step in the project involves loading the dance data for AI modeling. First, the preprocessed data is read in an interleaved manner to separately extract data from both dancers. Adjacencies are then created by initializing a default skeleton with 29 joints for each dancer and connecting every joint of one dancer to all joints of the other.

<div align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/models/assets/skeleton_and_connections.png", width="550">
</div>
<div align='center'>
    <span>Plot of the two dancers with only their skeletons (left) and fully connected to their partner (right).</span>
    <br><br>
</div>

The purpose of mapping both dancers' skeletons in this way is to ensure the model focuses on the connections between them, rather than the connections within each individual dancer. While a dancer's joint movements can often be predicted by analyzing their other joints, the aim here is to highlight the interactions between the dancers, identifying which joints of one dancer influence the other and vice versa. Additionally, both directions of the edges are considered, allowing the model to assess the direction of influence between each joint of the two dancers.

These connections are what the graph structure learning model will classify categorically by the degree of influence in the movement, ranging from non-existing to core connections. The data is then prepared for model training by creating batches with PyTorch tensors. The tensors are structured with dimensions representing the total number of sequences, the sequence length, the number of joints from both dancers, and 3D coordinates + 3D velocity estimates. Finally, a training-validation split is created to allow for proper model hyperparameter tuning.

<div align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/models/assets/velocity_estimation.png", width="550">
</div>
<div align='center'>
    <span>Plot of velocity estimations for the joints of both dancers (left and right). In each image, the arrows starts at the dancer's initial position and end at their position after a set number of frames, showing the velocity estimation for a given frame rate. In this experiment, a frame gap of 5 was used.</span>
    <br><br>
</div>

To include data augmentation and improve model generalization, the training pipeline incorporates a data processing step that involves rotating batches of data. Each batch is rotated along the Z-axis by a randomly selected angle while maintaining the original X and Y-axis orientations for physical consistency. This approach helps prevent the model from overfitting to the dancers' absolute positions.

Due to the high complexity of the problem, both in the number of moving particles and the number of edges in the graph, data simplification was implemented to achieve reasonable reconstruction performance and improve reliability during the sampling of influential edges. Random sampling of joints is applied, and only the subproblem of connecting the sampled joints is studied. This approach led to interesting results and a proof of concept that can be further explored to understand how to scale it.

<div align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/models/assets/sampled_dancers.png", width="550">
</div>
<div align='center'>
    <span>Plot showing an example of sampled joints, five for each dancer in this case. On the left, only the joints are displayed, while on the right, the joints are fully connected between the dancers (but not within a single dancer). Note that if there was already an edge connecting two joints within a dancer (from the skeleton), this edge is preserved as well.</span>
    <br><br>
</div>

## Neural Relational Inference Variant

As the title suggests, this model is a variant of the [Neural Relational Inference (NRI)](https://arxiv.org/abs/1802.04687) model, which itself is an extension of the traditional [Variational Autoencoder (VAE)](https://arxiv.org/abs/1312.6114). The primary objective of the original model is to study particles moving together in a system without prior knowledge of their underlying relationships. By analyzing their movement (position and velocity), the model estimates a graph structure that connects these particles, aiming to reveal which particles exert force on the others. For a dedicated study on the architecture's efficacy, refer to the [architecture experiment document](https://github.com/Luizerko/ai_choreo/blob/master/models/ARCH_EXPERIMENT.md), which discusses results based on a charged particles simulation dataset from the original paper.

In the context of this project, the particles are represented by the joints of dancers. While the physical connections between joints within a dancer's body are known, this information alone is insufficient to understand the partnering relationships between two dancers.

Since a target graph structure correctly identifying which joints are virtually connected during a dance performance is unavailable, and considering that this graph can change over time even within a performance, self-supervising techniques are employed - one of the reasons for choosing an autoencoder framework.

The model consists of an encoder and a decoder, both playing around with transforming node representations into edge representations and vice versa. This approach emphasizes the dynamics of movements rather than fixed node embeddings. Not only that, but the encoder specifically outputs edges, sampling these from the generated latent space, making it essential to switch between representations.

This project's implementation, even though very similar to the NRI MLP-Encoder MLP-Decoder model, includes a few important modifications:

- **Graph Convolutional Network (GCN):** Some Linear layers are replaced with GCN layers to leverage the graph structure, improving the model's ability to capture relationships between joints. This change focuses on a subset of edges connecting both dancers rather than studying all particle relationships as in the original implementation. Additionally, GCNs provide local feature aggregation and parameter sharing, important inductive biases for this context, resulting in enhanced generalization in a scenario with dynamic (and unknown) graph structures.

- **Graph Recurrent Neural Network (GRNN) Decoder:** To make better use of sequential information and achieve a more suitable final embedding for predicting (or reconstructing) the next frame, beyond just spatial information from the graphs, it is essential to use a recurrent network. The decoder is therefore implemented with LSTM nodes in the original sequence, while also using the graph structure sampled from the latent space generated by the encoder.

- **Custom GCN-LSTM Cells:** To utilize the recurrent structure crucial for sequence processing while maintaining graph information and GNN architecture, the classic LSTM cell has been reimplemented with GCN nodes. In the final version of the architecture, only the decoder incorporates the recurrent component, which generates a final sequence embedding that the model uses to reconstruct the next frame.

By incorporating these modifications, the model maintains the core principles of the original NRI model while theoretically enhancing its ability to generalize and adapt to the dynamic nature of dance performances.

<div align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/models/assets/final_arch.png", width="750">
</div>
<div align="center">
    <span>Schematic of the final model architecture, including the GCN nodes and the GRNN adapatation, inspired by the one found in the <a href='https://arxiv.org/abs/1802.04687'>original NRI paper</a>.</span>
    <br><br>
</div>

### Final Architecture

**Encoder:**

- The encoder begins with a GCN layer, which transforms node representations into edge representations.
  
- This is followed by a Linear layer, batch normalization, and dropout.

- Next, the edge representations are converted back into nodes, and another GCN layer is applied.

- The nodes are then transformed back into edges, followed by another Linear layer with a skip connection from the previous dropout layer.

- Finally, a Linear layer generates logits that represent edge types, ranging from non-existent edges to those most critical for the movement being analyzed.

**Decoder:**

- The decoder starts by sampling a Gumbel-Softmax distribution using the logits generated by the encoder. This approach approximates sampling in a continuous distribution and employs Softmax to handle the reparameterization trick, ensuring the pipeline remains fully differentiable.

- With the newly sampled edge index, the decoder processes the data through a GRNN composed of modified LSTM nodes with GCN layers, followed by a transformation of the final sequence embedding into edge representations.

- This is followed by a Linear layer, batch normalization, and dropout.

- Finally, the edge representations are converted back into nodes, and a GCN layer is applied to predict (or reconstruct) the next frame.

## Results

Getting this model to work was quite a challenge. The inherent complexity of Variational Autoencoders alone introduces numerous training difficulties. When combined with the complexity of the problem itself, the nuances of working on graph neural networks with dynamic graphs, and the task of generating a latent space that approximates a discrete distribution, it becomes a recipe for confusion. Understanding each part that needed adjustment, up to the challenge of training the neural network itself, was a long and involved journey. Still, in the end, some interesting results were achieved.

Given the subjective nature of the problem, it's hard to definitively evaluate the quality of the sampled edges. As such, the best analyses focus on the loss curves and the reconstructions obtained. The reasonableness of these two elements serves as a proxy for evaluating the sampled edges. In addition, some observations about patterns in edge sampling and a personal evaluation of the predicted relationships are included.

Due to the slow training process, the models discussed here were trained for 20 epochs. The resulting loss curves show healthy reconstruction error, and when looking at the validation dataset, there is no evidence of overfitting. However, the stagnation of the loss in the validation set suggests that the potential for significant model improvement with further training is uncertain. This could only be better explored by continuing training for more epochs.

On the other hand, the loss from the KL-Divergence is less nice than expected. It decreases sharply in the first few epochs and then plateaus. This is an unwanted effect, which can affect the quality of the latent space. Even though the phenomenon is reduced by training with beta coefficients, it is less effective in short training runs since these coefficients depend on the number of epochs.

<div align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/models/assets/final_loss_curve.png", width="500">
</div>
<div align="center">
    <span>Loss curve of one of the best-performing models. Six particles were sampled, three from each dancer, with data augmentation applied through dancer rotation, expanding the dataset by 10 times. The full encoder was used, with input sequences of 8 frames, hidden dimensions of 64 across all layers, and 4 types of edges. The model was trained using mean squared error for 20 epochs over 18 hours.</span>
    <br><br>
</div>

Regarding the reconstructions, there is significant variability. They are extremely sensitive to the sampled edges, which makes sense because, in a graph network, information spreads through neighboring nodes. If a joint is not connected to others by an edge, its reconstruction degenerates, and the particle stays stationary at the origin of the coordinate frame. It's common to find some of these in the reconstructions because the focus is on a specific category of edges among all edge types - those considered essential to the dancers' interactions. This limits the sampling, especially due to the prior distributions, which assumes the interaction being targeted is rare.

For the better reconstructions, it's clear that the particles are well positioned. The sampled dancers’ particles are positioned accurately in space, particularly in relation to the rest of the body. Additionally, the particles move in the same direction as the overall body movement, supporting good relative positioning. Given that the sampled connections only link particles between the dancers, it's impressive to see the valuable interaction captured in predicting their movement. It’s evident, however, that the best reconstructions occur when a particle has more than one connection, allowing information to propagate from a dancer to the other one and back through two hops.

Still, noticeable shaking in these particles is present and likely caused by two main factors: the inaccuracy in particle location and velocity in the original sequence, which already shows significant jitter, and the fact that the reconstructions are generated independently, although with the same sampled graph. This is because the best model version only predicts the next frame of a sequence. The impact of using this reconstructed frame as part of the input sequence to predict the next future frames in a chain was not explored.

|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/good_reconstruction_1.gif)|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/good_reconstruction_2.gif)|
|:-:|:-:|
<div align="center">
    <span></a>Here are two examples of better reconstructions: on the left, a scenario with more movement, and on the right, a more static scenario. The reconstructed sampled particles are colored differently for clarity, with purple for the blue dancer and orange for the red dancer.</span>
    <br><br>
</div>

Bad reconstructions unfortunately occur in several cases. First, as mentioned before, the reconstructions are highly sensitive to the sampled edges, and depending on the graph, they can be very poor, with the sampled joints barely moving. The model also struggles with sequences where the dancers switch sides, causing the reconstructed joints to remain in the initial position where the dancers started.

In more extreme cases, when the dancers move away from the center of the coordinate frame, the reconstructions tend to stay near the center. This issue comes from a normalization issue in the data processing pipeline that was identified later. The normalization of both dancers was removed to capture their relative movements, but a layer that normalized their combined movement was overlooked. As a result, the model faces more difficulty learning and dealing with these corner cases.

|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/stuck_particles.gif)|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/far_from_center.gif)|
|:-:|:-:|
<div align='center'>
    <span>Two examples of poor reconstructions: on the left, an example where edge sampling resulted in the reconstruction focusing on only 3 particles, despite 6 being sampled, while the other 3 remain static in the center of the coordinate frame. On the right, an example showing the model’s difficulty with reconstructions when the dancers are farther from the center, with the reconstructed particles either stationary or moving but inaccurately positioned.</span>
    <br><br>
</div>

Despite the challenges, the results achieved are still quite interesting. The loss curve shows a much more normal behavior, and the reconstruction results, while still facing significant issues, are becoming more reasonable. Given that each frame is reconstructed individually, making it difficult to maintain consistency between sequential frames, the particle positioning finally moved in the right direction. This made it possible to examine the learned edge distribution and the sampled edges across different examples to better understand how the model perceives the connections between particles, which, in turn, aids in reconstructing the frames.

It's also evident that the number of edges sampled with a confidence above 80% is consistently small, roughly matching the percentage of core edges in the prior distribution (both for 3 or 4 edge types). This indicates that the learned latent space indeed reflects the initial suggested distribution.

Now regarding the edges specifically, a few clear patterns emerge. First, among the core edges, most have the same confidence, with normally only one or two edges standing out less. This suggests a low hierarchy among the sampled edges, meaning that once an edge is part of the core group, it has a significant role in information propagation.

<div align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/models/assets/low_hierarchy_edges.gif", width="500">
</div>
<div align="center">
    <span>Example of the sampled edge distribution. The black edges represent connections between the dancers, with darker edges indicating higher confidence in their importance for reconstruction. In this typical case, for 6 particles sampled, 3 edges were selected, two with slightly higher importance, though all show a high confidence level (above 80%).</span>
    <br><br>
</div>

Additionally, it's common to find a particle connected by multiple edges. This likely relates to the earlier observation that information spreads more effectively within a dancer when there is a path that leaves and returns to the same particle within two hops.

|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/multi_edge_part_1.gif)|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/multi_edge_part_2.gif)|
|:-:|:-:|
<div align="center">
    <span>Examples of multiple edges connected to the same particle. On the left, an undirected example shows simpler movement for a sampling of 6 particles, while on the right, a directed example features more complex movement for a sampling of 10 particles. It is worth noting that, for the animation on the right, the model recognizes how one dancer's feet motion influence various parts of the other dancer's body, which aligns with the movement being performed, where the dynamic spin of the blue dancer guides the red dancer's response.</span>
    <br><br>
</div>

Another important pattern, though more subjective, is the connection between particles that are pending in opposition, like a string either being stretched or pulled between them. It seems that connections are more easily formed when particles are in this state of tension. This is intriguing because it reflects a key element of many movements in the choreography dataset, aligning with the project's core goal: to help dancers recognize the subtle dynamics in their partnering relationships.

|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/opposition_1.gif)|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/opposition_2.gif)|
|:-:|:-:|
<div align="center">
    <span>Examples of connections within opposition tendencies. On the left, the undirected example shows multiple connections between the lower torso of both dancers, first leaning in opposite directions and then gravitating toward each other, illustrating the full range of the stretched-string analogy. This example is notable for the model’s emphasis on opposition by sampling many edges for the reconstruction of these sequences. On the right, in the simpler directed example, one can see that the edges connecting particles moving apart are stronger than those linking particles moving closer together (as seen in the blue dancer's hand).</span>
    <br><br>
</div>

## Future of the Project

Despite the interesting results, there is clearly room for improvement. The main issues are the reconstructions, which remain unreliable and overly sensitive to the sampled edges, and the fact that only sampled joints are analyzed, not the dancers as a whole. However, there are many other challenges still to be addressed. Considering all of this, several next steps are suggested for further development of this project:

- **Data collection**: Regardless of the use of data augmentation through duet rotation, the small dataset size made it difficult to train a complex network, especially given the difficulty of the problem.

- **Data quality**: The pipeline used to extract 3D poses from video, though functional, has room for improvement. Even in the original sequences, the dancers’ particles are shaky and often poorly approximated, leading to extreme and random movements (unrealistic). Moreover, normalization of both dancers was removed to preserve relative movement, but it was realized too late that a new layer normalizing their joint movement should have been added. Having dancers in different parts of space caused confusion for the model.

- **Architecture exploration**: While **many** versions of the NRI variant were implemented and tested, the final version is still far from being ideal, and there hasn't been a true breakthrough moment. There is still significant potential in this architecture, especially given the strong results achieved by the original version.

- **Processing speed**: Several parts of the final architecture are custom implementations, so they are far from optimized. Batches are replaced with sequential operations at several points in the pipeline: in transformations between node and edge representations, in the decoder, since a new set of edges is sampled for each sequence in a batch, and in the GRNN due to its sequential nature. As a result, a training cycle with just a few dozen epochs can take an entire day. Optimizing this process is crucial to the project scaling.

- **Interaction with dancers**: Since this project sits at the intersection of art and technology, direct interaction with artists is essential. In a more refined version of the model with consistent results, it would be ideal to present the tool to the dance community, share its concept, demonstrate the results, and observe how dancers use the tool in their own partnering studies.