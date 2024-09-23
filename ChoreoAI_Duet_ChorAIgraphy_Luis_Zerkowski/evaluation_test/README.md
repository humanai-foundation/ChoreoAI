# Evaluation Test Report

This is a file containing all the information regarding the development of the evaluation test.

## Setup

Before using the notebook, you should setup your environment:

```
conda create -n <NAME_OF_YOUR_ENVIRONMENT> python=3.10
```

After creating the conda environment, just install all the dependencies by running:

```
pip install -r requirements.txt
```

Alternatively, if the dependency versions don't match your system (very likely because of the CUDA version), you can manually install all the core packages found in `requirements.in`. They should also require you to install compatible versions of all the packages contained in `requirements.txt`.

## Loading Data

This is the pre-test part of the project that consists of replicating [Mariel's code](https://github.com/mariel-pettee/choreo-graph/blob/main/functions/load_data.py) to properly load and preprocess the [provided data](https://github.com/mariel-pettee/choreo-graph/tree/main/data). In here you can find code to load data, put everything in a very handable data structure and format, preprocess the joint positions so that they belong to the same unit cube (since we are interested in relative motion instead of absolute motion), and finally compute the edges.

## Visualizing Dance

This is the first part of the test, in which I effectively started developing. In here I instantiated the `MarielDataset` class, experimented for quite a while with the data to understand what each part actually represented, then built up a static visualization scheme to make sure everything was in order and finally animated a sequence from the original dataset.

Note: I did not include the experimentation parts to this section of the notebook because I didn't want to make it even longer.

<br>

<p align="center">
  <figure>
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/evaluation_test/assets/visualizing_sequence.gif" alt="Visualizing an Original Sequence" width="400">
    <figcaption align="center">Visualizing a sequence from the original dataset.</figcaption>
  </figure>
</p>

## Training Generative Model

This is the second part of the test and the most difficult one. To make all the descriptions more clear, I separate them into different sections:

 ### Implementation
 
 I decided to go for the LSTM-VAE model. This decision was based in two main reasons:

- I wanted to replicate the ideas used in the [provided paper](https://arxiv.org/pdf/1907.05297.pdf). I understood that they resulted in a good model as described in the paper and also that I could try to use the given hyperparameters, making the search space for optimization much easier. Since optimizing NNs can often prove to be quite a challenge, I thought it would be a good idea considering the time schedule.

- I have much more experience with LSTM than with GNNs, so I thought I should stick to models I'm more familiar with because of time limitations. I figured I could explore more about GNN models within the development of the real project if I get accepted.

### Architecture and Optimization

- One encoder with 3 LSTM layers (384 nodes) and 2 separated branches of linear layers (256 nodes each for the latent space), one for the mean and another one for log-variance.

- One decoder with 1 linear layer (384 nodes) with ReLU activation function for the latent-space sampled data and 3 LSTM layers (159 nodes for the output).

The model was trained with Adam optimizer for 200 epochs with early stopping at 3 validation losses higher than the best validation loss at that point. I also used the KL-divergence weight provided in the paper (0.0001). Finally, I added 0.2 dropout for the LSTM layers for some more regularization.

I expanded the dataset using data noise augmentation. I got around 10000 random sequences out of the almost 40000 provided sequences, and added 0.01 scaled Gaussian noise to the joint coordinates to try and make the model a bit more robust. I wanted to do even more augmentation, but due to my GPU limitations, this was the best I could do. Finally, I used 90% of the data for training  and 10% for validation, both randomly sampled from the dataset and shuffled afterwards, with a batch sizes of 64.

### Comments and Results

Even though I had reduced a lot the hyperparamter space by trying to replicate the provided paper, I still ended up having to train the model multiple times to find out the best hyperparameters.

Furthermore, I had issues with the validation loss that made me replicate the experiments an enormous amount of times. I had a decreasing validation loss, as expected, but still orders of magnitude larger than the training loss. I think this problem is mostly related to the model being a bit to complex for the amount of data I had ($\frac{2}{3}$ of the data size from the paper with much less augmentation due to GPU limitations). I tried reducing the amount of LSTM layers to make the model simpler, but found out it was not really capable of capturing the complexity of dance sequences, mostly generating sequences in which the figure stands almost still. The image below shows the loss curves for training and validation datasets throughout the training process:

<br>

  <figure align="center">
    <img src="https://github.com/Luizerko/ai_choreo/blob/master/evaluation_test/assets/loss_graphs.png" alt="Loss Curves" width="600">
  </figure>

<br>

A better solution I could think of was to reduce the sequence length drastically (64 instead of 128) to both expand the dataset and make the sequences much more simple to learn. It indeed reduced the validation loss quite a lot, but generated very bad sequences (almost random joint positions and movement). In the end I did not have the time to properly evaluate all the hyperparameters possibilities for this reduced sequence model and went back to the original sequence lengths implementation that had much better results at least.

Even with all these issues, I managed to train the model and come up with some very interesting results. Some sequences from the original dataset are accurately reconstructed by the model. Others, even if not perfectly reconstructed, still clearly show that the model was able to capture the essence of their movements. A sequence that rotates, for example, remains rotating in its reconstructed version, or a sequence that lifts its leg remains with this movement in the reconstruction as well.

When it comes to generating new sequences, the model is quite sensitive to the standard deviation used. When the latent space is sampled with a normal distribution, the model generates interesting sequences, but with fewer movements than the original sequences. When the latent space is sampled with a higher standard deviation, the sequence tends to be more creative, but it is also common to see joints getting lost in space (many points converging to the same coordinates or points moving shakily).

Finally, one behavior I did not manage to fix was the initial state of the joints. Even in the best reconstructed/generated sequences, the joints start in weird positions, making the first miliseconds of the animation almost glitch to proper positions and then start a proper sequence of movements.

In the GIFs below I show some of the obtained results.

<br>

<table>
  <tr>
    <td>
      <img src="https://github.com/Luizerko/ai_choreo/blob/master/evaluation_test/assets/original_seq.gif" alt="Original Sequence" width="400" /><br>
      <figcaption align="center">Orignal sequence.</figcaption>
    </td>
    <td>
      <img src="https://github.com/Luizerko/ai_choreo/blob/master/evaluation_test/assets/recon_seq.gif" alt="Reconstructed Sequence" width="400"/><br>
      <figcaption align="center">Reconstructed sequence.</figcaption>
    </td>
  </tr>
</table>

<br>
<br>

<table>
  <tr>
    <td>
      <img src="https://github.com/Luizerko/ai_choreo/blob/master/evaluation_test/assets/bad_original_seq.gif" alt="Bad Original Sequence" width="400" /><br>
      <figcaption align="center">Original sequence that doesn't reconstruct well.</figcaption>
    </td>
    <td>
      <img src="https://github.com/Luizerko/ai_choreo/blob/master/evaluation_test/assets/bad_recon_seq.gif" alt="Bad Reconstructed Sequence" width="400"/><br>
      <figcaption align="center">Not so good reconstructed sequence. The core movements are captured by the reconstruction though.</figcaption>
    </td>
  </tr>
</table>

<br>
<br>

<table>
  <tr>
    <td>
      <img src="https://github.com/Luizerko/ai_choreo/blob/master/evaluation_test/assets/generated_seq.gif" alt="Generated Sequence" width="400" /><br>
      <figcaption align="center">Generated sequence.</figcaption>
    </td>
    <td>
      <img src="https://github.com/Luizerko/ai_choreo/blob/master/evaluation_test/assets/generated_seq_me.gif" alt="Interpretation of Generated Sequence" width="400"/><br>
      <figcaption align="center">My interpretation of the generated sequence.</figcaption>
    </td>
  </tr>
</table>

<br>
<br>

<table>
  <tr>
    <td>
      <img src="https://github.com/Luizerko/ai_choreo/blob/master/evaluation_test/assets/generated_seq2.gif" alt="Another Generated Sequence" width="400" /><br>
      <figcaption align="center">Another generated sequence.</figcaption>
    </td>
    <td>
      <img src="https://github.com/Luizerko/ai_choreo/blob/master/evaluation_test/assets/generated_seq2_me.gif" alt="Interpretation of Another Generated Sequence" width="400"/><br>
      <figcaption align="center">My interpretation of another generated sequence.</figcaption>
    </td>
  </tr>
</table>

<br>
<br>

<table>
  <tr>
    <td>
      <img src="https://github.com/Luizerko/ai_choreo/blob/master/evaluation_test/assets/generated_seq_shaky.gif" alt="Shaky Generated Sequence" width="400" /><br>
      <figcaption align="center">Generated sequence with a bit larger standard deviation ($1.5\sigma^2$).</figcaption>
    </td>
    <td>
      <img src="https://github.com/Luizerko/ai_choreo/blob/master/evaluation_test/assets/generated_seq_2std.gif" alt="2 Standard Deviation Generated Sequence" width="400"/><br>
      <figcaption align="center">Generated sequence with even larger standard deviation ($2\sigma^2$).</figcaption>
    </td>
  </tr>
</table>


## Why This Project?

Growing up in the northeast of Brazil, art and communication were central to my life. Surrounded by the richness of Brazilian music and dance from a young age, I quickly connected to the arts. While I'm not as skilled as many dancers, I strongly believe in the power of dance to bring warmth to any environment and I always engage in it with joy. My talents, though, are more related to communication, making me a natural-born chatterbox, always wanting to learn from others, and share parts of my own journey. I think the mixture of my cultural background along with my academic and professional trajectory is exactly what connects me to this project and truly makes me want to be a part of it.

Now talking about approaches to the project, I would first focus on building the dataset. I would use the same methods employed in the provided paper to preprocess the dance sequences, but now separating the joints of the two dancers into different groups and encoding the interaction between nodes from the two. Focusing on the proposed methods, we could:

- Draw and edge between joints from different dancers that have some common properties. Very close nodes with either same or opposite velocity vectors, and nodes in symmetric positions are examples of what these properties could be. This form of encoding fits perfectly a GNN and could be used to help the model understand the sequences while capturing the relationship between dancers.

- Use the same approach for pairs of nodes computation as before, but rather than encoding pairs with an edge, encoding them into a dance sequence by a special connection token. These tokens could be used by a transformer model to capture the relationship between nodes over time, understanding a sequence as combination of interactive joint pairs.