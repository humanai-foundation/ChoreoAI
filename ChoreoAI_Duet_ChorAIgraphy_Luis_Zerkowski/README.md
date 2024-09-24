# AI-Generated Choreography - from Solos to Duets

This repository is dedicated to the development of the project [AI-Generated Choreography - from Solos to Duets](https://humanai.foundation/gsoc/2024/proposal_ChoreoAI1.html). Here, you will find all the documentation and code for implementing my pipelines.

Special thanks to my supervisors, [Mariel Pettee](https://marielpettee.com/) and [Ilya Vidrin](https://www.ilyavidrin.com/), for all their guidance, and to my work partner, Zixuan Wang, for developing [her pipeline](https://github.com/humanai-foundation/ChoreoAI/tree/main/ChoreoAI_Zixuan_Wang) alongside mine.

If you don't want to dive into the code right away but rather want to have an overview of the entire project, check my [blog posts on Medium](https://medium.com/@luisvz).

## Duet ChorAIgraphy: Using GNNs to Study Duets

<div align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/assets/duet_choraigraphy_logo.png", width="450">
</div>

Duet ChorAIgraphy aims to implement a pipeline using Graph Neural Networks (GNNs) to study dance duets. The project focuses on **Interpretability of Movements:** a pipeline that learns about the connection between the dancers' bodies in different dance sequences.

<!-- 2. **Generation of New Sequences:** A pipeline that uses these learned connections to generate new dance sequences. -->

The pipeline is discussed in more detail below, along with a presentation of key results. For a more comprehensive explanation of the model, refer to the `models` directory, which contains its implementations as well as complete documentation on it.

## Repository Hierarchy

- The root of the repository naturally provides an overview of the project and showcases sample generations from the developed model.

- The `evaluation_test` folder contains the application for the contributor position. It includes a `README` file with detailed information on the implementations for the selection process test, along with the results obtained. The folder also contains the notebook with the source code for the developed components. The `README` was later expanded as the project development continued.

- The `pose_extraction` folder contains details about the pose extraction pipeline used to create the project dataset. It includes an installation process guide, the raw data used, and information on the development and execution of the idea.

- The `models` folder contains the project's core, including installation instructions, pipelines designs, and running guidelines for the interpretability of movements model. It also has detailed results for a thorough evaluation on the quality of the final agent.

## Sample Generations

The project has produced a wide range of results. Its creative and subjective nature offers many opportunities for exploration, even in the most unexpected outputs. However, while this subjectivity allows for different perspectives, it also creates challenges in getting a more precise evaluation of the model. Below are examples of connections (both undirected and directed) computed by the model in various scenarios. For a deeper study about what the model is doing, please check the [documentation under `models`](https://github.com/Luizerko/ai_choreo/blob/master/models/README.md).

<div align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/models/assets/low_hierarchy_edges.gif", width="500">
</div>
<div align="center">
    <span>Example of the sampled edge distribution. The black edges represent connections between the dancers, with darker edges indicating higher confidence in their importance for reconstruction. In this typical case, for 6 particles sampled, 3 edges were selected, two with slightly higher importance, though all show a high confidence level (above 80%).</span>
    <br><br>
</div>

|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/multi_edge_part_1.gif)|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/multi_edge_part_2.gif)|
|:-:|:-:|
<div align="center">
    <span>Examples of multiple edges connected to the same particle. On the left, an undirected example shows simpler movement for a sampling of 6 particles, while on the right, a directed example features more complex movement for a sampling of 10 particles. It is worth noting that, for the animation on the right, the model recognizes how one dancer's feet motion influence various parts of the other dancer's body, which aligns with the movement being performed, where the dynamic spin of the blue dancer guides the red dancer's response.</span>
    <br><br>
</div>

|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/opposition_1.gif)|![](https://github.com/Luizerko/ai_choreo/blob/master/models/assets/opposition_2.gif)|
|:-:|:-:|
<div align="center">
    <span>Examples of connections within opposition tendencies. On the left, the undirected example shows multiple connections between the lower torso of both dancers, first leaning in opposite directions and then gravitating toward each other, illustrating the full range of the stretched-string analogy. This example is notable for the model’s emphasis on opposition by sampling many edges for the reconstruction of these sequences. On the right, in the simpler directed example, one can see that the edges connecting particles moving apart are stronger than those linking particles moving closer together (as seen in the blue dancer's hand).</span>
    <br><br>
</div>

|![](https://github.com/Luizerko/ai_choreo/blob/master/assets/beauty_1.gif)|![](https://github.com/Luizerko/ai_choreo/blob/master/assets/beauty_2.gif)|
|:-:|:-:|
<div align="center">
    <span>Two final examples are presented not for specific details, but to highlight and appreciate interesting visualizations generated by the model.</span>
    <br><br>
</div>

For a quick and easy way to generate and explore new visualizations, check out the [Colab Notebook](https://colab.research.google.com/drive/1KhX-Ppn9-BxAO4EX0BtfqPohdA09I6z5?usp=sharing).
