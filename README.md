# Seq2Seq Model with Attention for Sequence Prediction

This project implements a sequence-to-sequence (Seq2Seq) model with attention using PyTorch. The model is designed for tasks involving sequential data, such as machine translation, text summarization, and speech recognition.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Model Architecture](#model-architecture)
4. [Dependencies](#dependencies)
5. [Getting Started](#getting-started)
6. [Usage](#usage)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Contributing](#contributing)
10. [License](#license)

---

## Introduction

Seq2Seq models map an input sequence to an output sequence, often of different lengths. This implementation includes attention mechanisms, enhancing the model's ability to handle longer input sequences by focusing on relevant parts of the input during decoding.

---

## Features

- **Bidirectional Encoder**: Captures context from both directions of the input sequence.
- **Attention Mechanism**: Dynamically focuses on parts of the input sequence during decoding.
- **Teacher Forcing**: Improves training efficiency and convergence.
- **Modular Design**: Separate modules for the encoder, decoder, and attention.
- **Flexible Hyperparameters**: Easily configurable model parameters.

---

## Model Architecture

The project consists of three main components:
1. **Encoder**: Processes the input sequence and generates context vectors.
2. **Attention**: Computes alignment scores between the decoder's hidden state and encoder outputs.
3. **Decoder**: Generates the output sequence step-by-step using the context vectors and its own hidden states.

---

## Dependencies

This project requires Python and the following packages:
- `torch` (PyTorch)
- `numpy`
- `random`

To install dependencies, run:

    pip install torch numpy


## Getting Started

### Clone the Repository

      git clone <repository-url>
      cd <repository-folder>


### Prepare the Dataset

Ensure your dataset is in the required format (input and target sequences).

---

## Usage

- **Encoder**: The encoder processes the input sequence and generates a set of outputs and hidden states.

      encoder = Encoder(input_size, embed_size, hidden_size, n_layers, dropout)
      encoder_outputs, hidden = encoder(input_sequence)

- **Decoder**: The decoder generates the output sequence one step at a time.

      decoder = Decoder(embed_size, hidden_size, output_size, n_layers, dropout)
      output, hidden, attn_weights = decoder(input_step, last_hidden, encoder_outputs)


- **Full Seq2Seq Model**: Combine the encoder and decoder for end-to-end training.

      seq2seq = Seq2Seq(encoder, decoder)
      outputs = seq2seq(src, trg, teacher_forcing_ratio)


---

## Training

Use the `train.ipynb` notebook to train the model. Key steps include:

- Loading the dataset.
- Initializing the model and optimizer.
- Running the training loop with teacher forcing.

### Sample Training Loop

```bash
for epoch in range(num_epochs):
loss = 0
for batch in dataloader:
src, trg = batch
optimizer.zero_grad()
output = model(src, trg, teacher_forcing_ratio=0.5)
batch_loss = loss_fn(output, trg)
batch_loss.backward()
optimizer.step()
loss += batch_loss.item()
print(f"Epoch {epoch+1}, Loss: {loss/len(dataloader)}")
```


---
