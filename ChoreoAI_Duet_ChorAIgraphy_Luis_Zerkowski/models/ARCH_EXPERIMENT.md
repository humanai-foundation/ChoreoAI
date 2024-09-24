# Addressing a Simpler Problem

During the development of the model for the dance scenario, challenges arose in reconstructing frames and predicting edges. These issues, along with extensive model iterations, data augmentations, and various experiments, led to questioning the architecture's ability to learn the desired dynamics, prompting further testing.

To better assess the model's capabilities, the decision was made to revisit the [original NRI paper](https://arxiv.org/abs/1802.04687) and use their charged particles dataset. The original problem was much simpler — featuring ten times fewer particles, hundreds of times fewer edges, a 2D environment and a greater availability of data — providing a clearer benchmark for evaluating the NRI variant's performance on a less complex task.

The experiment provided valuable insights. Even in a simplified version with significantly reduced hidden dimensions, the model showed the ability to produce reasonable reconstructions and edge predictions. Although the results are not perfect, the model's potential to address the graph dynamics problem became clear.

These findings have shifted the focus to refining the architecture and adjusting parameters within the simpler problem's framework. The goal is to identify a version that can be scaled up to effectively tackle the more complex dance setting. Simultaneously, the goal is to simplify the choreography dataset by selectively sampling dancers, aligning both experiments more closely and better exploiting the architecture.

## Generating Data

The data generation process involved using the `data/generate_dataset.py` script with the default settings from the [original project repository](https://github.com/ethanfetaya/NRI), but with the option to simulate charged particles in motion. This resulted in a training dataset of 50000 simulations, each simulation including 5 particles with 2D coordinates and 2D velocities, moving over 49 frames. The interactions between charges are defined by a random undirected graph, which controls the particles' accelerations, affecting their speed and position in each frame.

<div align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/models/assets/trajectories_and_graph.png", width="600">
</div>
<div align='center'>
    <span>On the left, you can see a complete simulation, where the paths of different particles are shown in various colors, starting their trajectories with lighter shades and ending with darker ones. On the right, you see the interaction graph between the particles, where the edges between vertices indicate bidirectional interactions between the respective particles.</span>
    <br><br>
</div>

To prepare the model input, the original sequence of 49 frames is first reduced to 48 frames to simplify testing different sequence splits. Subdivisions are then tested using sub-trajectories of 8, 12, 16, or 24 frames. To ensure the training process remains within a reasonable time frame (under 24 hours), non-overlapping sequences are used, meaning each simulation is divided into non-overlapping segments of $\frac{48}{sequence\_size}$. The next section provides additional visualizations and detailed information on the performance of the models tested for each sequence size.

In addition, the input needs to be prepared from the graph's perspective. The approach, similar to the original choreography model, involves using a fully connected undirected graph. This allows the model to later sample edges based on the input sequence and retain only those relevant for reconstructing the trajectory. The image below illustrates an example of the input graph from a random frame in a random sequence.

<div align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/models/assets/fully_connected_graph.png", width="300">
</div>
<div align='center'>
    <span>The image shows a fully connected graph used as the model's input. In this example, a random frame from a simulation is chosen, and all points are connected.</span>
    <br><br>
</div>

## Testing Architecture

Using samples from the data described above, different architectures were tested to assess the model's ability to reconstruct sequences. In an autoencoder and for a self-supervised task, it's important not only to look at loss curves but also to observe the model's outputs. In this type of architecture, and particularly in a subjective context like duet choreography, the sampled edges in the latent space may seem reasonable even when the model is completely off track. Since the reconstruction heavily relies on the sampled edges, evaluating the quality of the reconstructions helps to better judge the quality of the interactions suggested by those edges.

This effect is evident in some experimental results shown below. Usually, when a particle has no sampled interactions, it stays still in the center. Also, particle movement is naturally linked to the particles they interact with, so good edge sampling is crucial for accurate reconstructions.

<div align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/models/assets/mov_part_edges.png", width="350">
</div>
<div align='center'>
    <span>The image illustrates the reconstruction of a 6-frame sequence, where only the particles connected by an edge are in motion. This edge sample and reconstruction come from a trained compact architecture model.</span>
    <br><br>
</div>

For the following experiments, these were the original trajectories and graph considered:

|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/original_trajectories_6seq.png)|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/original_trajectories_12seq.png)|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/original_graph.png)|
|:-:|:-:|:-:|

<div align='center'>
    <span>On the left, the trajectory for the 6-frame sequence; in the middle, the trajectory for the 12-frame sequence; and on the right, the interaction graph for both trajectories.</span>
    <br><br>
</div>

### Compact Architecture

This architecture is a simplified version of the originally implemented encoder. The goal was to reduce the complexity of the architecture to see if the model could still capture the relationships between particles and to determine if the problem could be solved with fewer data transformations.

Key differences between this model and the original:

- It uses only one transformation from nodes to edges representation.
- It removes an MLP layer, dropout, and batch normalization (which were part of the processing pipeline for the first transformation from nodes to edges).
- Skip connections are not used due to the shorter network.

For this architecture, four models were trained, and their example reconstructions and loss curves are presented below:

1. **Model with 6-frame sequences**

|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/lc_small_6seq.png)|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/rec_small_6seq.png)|
|:-:|:-:|

<div align='center'>
    <span>In the left image, the reconstruction loss curve looks healthy, showing positive signs for continuing training the model. In the right image, the reconstruction is quite interesting. While it doesn't fully capture the movement of all the particles, the model does a good job of accurately locating the particles, especially in their relative positions to each other, and capturing some of the direction and shape of the movement. Not all particles behave this way - like the red particle, which stayed still in the center - but this is still a big improvement compared to earlier results.</span>
    <br><br>
</div>

2. **Model with 6-frame sequences, but with more edge types (4 instead of binary)**

|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/lc_small_6seq_4edges.png)|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/rec_small_6seq_4edges.png)|
|:-:|:-:|

<div align='center'>
    <span>Once again, the left image shows healthy reconstruction loss curves, although the validation loss is a bit more unstable. As for the reconstruction itself, this was likely one of the best results. The model almost perfectly captured the movement and location of three particles (green, black, and blue) and managed to get the shape of the movement for another particle (orange), though the location wasn't as accurate. The remaining particle (red) was reasonably well-located but had no movement captured. This promising result supports the idea of using more edge types, allowing the model to better determine the strength of interactions and improve reconstructions.</span>
    <br><br>
</div>

3. **Model with 12-frame sequences**

|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/lc_small_12seq.png)|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/rec_small_12seq.png)|
|:-:|:-:|

<div align='center'>
    <span>Although the reconstruction loss curves for this model, shown on the left, are still reasonable, the reconstruction on the right reveals more significant shortcomings compared to previous models. Not only do more particles fail to move (both the orange and red particles remain stationary in the center), but the movement of the captured particles quickly deteriorates after a certain number of frames. For instance, the green and black particles, which start off well-reconstructed, lose their trajectory completely after the first 6 predicted frames. The model shows a clear limitation in maintaining coherence over extended sequences. Additionally, the movement of the blue particle is reconstructed in the opposite direction of its actual movement. It becomes clear that, for this compact model, longer motion sequences are much more difficult to interpret.</span>
    <br><br>
</div>

4. **Model with 12-frame sequences with recurrent encoder**

|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/lc_small_12seq_recurrent.png)|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/rec_small_12seq_recurrent.png)|
|:-:|:-:|

<div align='center'>
    <span>To reduce the hallucination effects in long sequences seen with the previous model, a recurrent encoder was introduced. However, the results show reconstruction loss curves (left) very similar to the previous one, and the reconstruction quality (right) might even be worse. For example, the black particle's movement is incorrectly reconstructed from the first frame. This suggests that the compact version of the model is really not able to handle the complexity of longer sequences. Despite this, there is a possible qualitative improvement in intra-sequence coherence thanks to the recurrent encoder. The movements of the black and blue particles were reconstructed more smoothly, and the green particle showed a less severe deviation compared to earlier results.</span>
    <br><br>
</div>

### Standard Architecture

This is the originally implemented architecture, without any modifications to the one already introduced in the [README](https://github.com/Luizerko/ai_choreo/blob/master/models/README.md), only varying the length of the input sequences used.

1. **Model with 6-frame sequences**

|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/lc_standard_6seq.png)|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/rec_standard_6seq.png)|
|:-:|:-:|

<div align='center'>
    <span>Once again, the left image shows healthy reconstruction loss curves, indicating the potential for continuing the training process. On the right, there is an intriguing reconstruction. Despite the visual impact caused by the scale - since the green and blue particles did not have their movements predicted - the model provides a much more realistic reconstruction for the orange and red particles. In these cases, both the location and the movement shape are well predicted, with an interesting anticipation of their speeds, as if the future movement was predicted. For the black particle, the approximation of both location and movement is somewhat rough but still reasonable.</span>
    <br><br>
</div>

2. **Model with 12-frame sequences**

|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/lc_standard_12seq.png)|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/rec_standard_12seq.png)|
|:-:|:-:|

<div align='center'>
    <span>In this case, even though the left image shows healthy reconstruction loss curves, indicating the potential for continuing the training process, the reconstruction on the right is entirely ineffective. The model trained on 6-frame sequences already showed some difficulty, possibly indicating the need for more data, but the 12-frame sequence model failed completely. Three particles (red, blue, and black) had no movement captured, and the two that did (green and orange) exhibited almost random, uncoordinated, and incoherent movement.</span>
    <br><br>
</div>

### Big Architecture

Similar to the previous architecture, this one is the same as the original, with the only difference being an increase in the hidden dimensions from 32 to 64. This change was made to give the model more capacity to handle complex adaptations.

1. **Model with 6-frame sequences**

|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/lc_big_6seq.png)|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/rec_big_6seq.png)|
|:-:|:-:|

<div align='center'>
    <span>This model continues to show healthy reconstruction loss curves (left) but struggles with reconstruction (right). Two particles (black and blue) had no movement captured, while the remaining three (green, orange, and red) displayed poor movement predictions, despite being well-localized. Overall, it seems that more data and training time might be needed for this model to produce better predictions.</span>
    <br><br>
</div>

2. **Model with 12-frame sequences**

<div align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/models/assets/rec_big_12seq.png", width="400">
</div>
<div align='center'>
    <span>The image shows the reconstruction generated by this model, which demonstrates decent particle localization but fails to provide coherent movement predictions. This issue becomes more pronounced with longer sequences.</span>
    <br><br>
</div>

### Conclusion

In the end, it becomes clearer that the implemented architecture shows potential for addressing such a task effectively. In a simplified scenario with more abundant data, it was observed that after just a few iterations (between 10 and 20 epochs, depending on the model's training time), the reconstruction loss curves displayed typical and more controlled behavior. Furthermore, these curves mostly indicated the model's potential for further improvement with more iterations, and no signs of overfitting yet.

The rapid convergence of the KL Divergence is attributed to a slightly high beta coefficient during the autoencoder training process. Since beta was adjusted based on the number of epochs, when the model was trained for fewer epochs, beta increased more quickly. Thus this kind of behavior in the KL loss was expected. Although not ideal, the quick convergence did not lead to one of its common side effects: the failure to optimize reconstruction error, which allowed for smooth training overall.

Additionally, it became clear that the more simplified models, particularly those using shorter frame sequences and giving the architecture more freedom to choose the impact of an edge (more edge types), produced some promising reconstruction results. Naturally these are the versions carried out to the original duet setting for training.