# AI-Generated Choreography - from Solos to Duets

This repository is dedicated to the development of the project [AI-Generated Choreography - from Solos to Duets](https://humanai.foundation/gsoc/2024/proposal_ChoreoAI1.html). Here, you will find all the documentation and code for implementing my pipelines.

Special thanks to my supervisors, [Mariel Pettee](https://marielpettee.com/) and [Ilya Vidrin](https://www.ilyavidrin.com/), for all their guidance, and to my work partner, Zixuan Wang, for developing [her pipeline](https://github.com/wang-zixuan/AI-Choreo-Duets/tree/main) alongside mine.

## Duet ChorAIgraphy: Using GNNs to Study Duets

<div align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/assets/duet_choraigraphy_logo.png", width="450">
</div>

Duet ChorAIgraphy aims to implement two pipelines using Graph Neural Networks (GNNs) to study dance duets. The project focuses on two main aspects: 

1. **Interpretability of Movements:** A pipeline that learns about the connection between the dancers' bodies in different dance sequences.

2. **Generation of New Sequences:** A pipeline that uses these learned connections to generate new dance sequences.

Below, I discuss each of these pipelines in more detail and present some results. For a more comprehensive explanation of the models, I encourage you to check the `models` directory, which contains their implementations as well as more complete documentation on both.

## Repository Hierarchy

- The root of the repository naturally provides an overview of the project and showcases sample generations from the developed models.

- The `evaluation_test` folder contains my application for the contributor position. In there, you can find a `README` with all the information regarding the implementations for the selection process test and also the results obtained. You can also find the notebook with the source code for everything I developed. Fortunately, I had the pleasure to expand this `README` and continue the project development.

- The `pose_extraction` folder contains details about the pose extraction pipeline used to create the project dataset. It includes an installation process guide, the raw data used, and information on the development and execution of the idea.

- The `models` folder contains the project's core, including installation instructions, pipelines designs, and running guidelines for both the interpretability of movements and the choreography generation models. It also has detailed results for a thorough evaluation on the quality of the final agents.