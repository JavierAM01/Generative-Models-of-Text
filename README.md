# Generative Models of Text

### Overview
This project explores generative text models, focusing on Recurrent Neural Networks (RNNs) and Transformer-based language models. It involves implementing and experimenting with architectures such as bidirectional RNNs, Transformers, Sliding Window Attention, Rotary Positional Embeddings (RoPE), and Grouped Query Attention (GQA). The goal is to understand how different generative techniques impact language modeling and computational efficiency.

## Project Structure
```
├── requirements.txt
├── input.txt
├── chargpt.py
├── mingpt/
│   ├── model.py
│   ├── trainer.py
│   ├── utils.py
├── test_model.py
├── README.md
```

## Getting Started

### Installation
To set up the environment locally, follow these steps:
1. Install Python dependencies:
   ```sh
   pip install torch einops
   ```
2. Run the main training script:
   ```sh
   python chargpt.py
   ```

## Recurrent Neural Network (RNN) Language Models

### Model Implementation
The RNN language model uses an Elman Network with hidden states updated as follows:
```
ht = slide(Whh * ht-1 + Whx * xt + bh)
yt = slide(Wyh * ht + by)
```
where `slide(a) = min(1, max(0, a))` ensures values stay within a fixed range. The model processes sequential text data, capturing dependencies over time.

### Experiments
- Implemented a simple RNN-based language model.
- Explored bidirectional RNNs and their inability to serve as autoregressive models.
- Compared how different architectures handle sequential context.

<img width="900" height="250" src="https://github.com/JavierAM01/Generative-Models-of-Text/blob/main/images/rnn.png">

## Transformer Language Models

### Model Implementation
The Transformer model is based on scaled dot-product attention:
```
st,j = kT * qt / sqrt(|k|)
at = softmax(st)
```
where queries, keys, and values are computed as:
```
vj = Wv * xj, qj = Wq * xj, kj = Wk * xj
```
We also analyzed alternative attention mechanisms, such as multiplicative and additive attention.

### Experiments
- Implemented scaled dot-product attention.
- Explored multiplicative attention and its impact on model expressiveness.
- Analyzed self-attention properties, including conditions for symmetry.

<img width="900" height="500" src="https://github.com/JavierAM01/Generative-Models-of-Text/blob/main/images/attention.png">

## Sliding Window Attention
Sliding Window Attention improves efficiency by restricting context to a fixed window size `w`, rather than attending to the entire sequence.

### Implementation
- Defined causal masks for attention computation.
- Optimized time complexity from `O(N^2)` to `O(Nw)`.
- Reduced space complexity from `O(N^2)` to `O(N + w)`.

### Experiments
- Implemented optimized Sliding Window Attention.
- Evaluated computational efficiency against naive matrix multiplication.

<img width="900" height="500" src="https://github.com/JavierAM01/Generative-Models-of-Text/blob/main/images/window.png">

## Rotary Position Embeddings (RoPE)
RoPE encodes relative positional information directly into the attention mechanism, replacing absolute position embeddings.

### Implementation
- Implemented RoPE in the `RotaryPositionalEmbeddings` class.
- Modified the `CausalSelfAttention` class to integrate RoPE embeddings.

### Experiments
- Compared text samples generated with and without RoPE.
- Evaluated training loss across different training schedules.

<img width="900" height="500" src="https://github.com/JavierAM01/Generative-Models-of-Text/blob/main/images/rope_2.png">

## Grouped Query Attention (GQA)
GQA reduces memory usage by sharing key-value pairs across query groups, balancing efficiency and performance.

### Implementation
- Implemented `GroupedQueryAttention` in `model.py`.
- Modified the attention mechanism to support grouped query heads.

### Experiments
- Measured attention computation time across different numbers of key heads.
- Compared training loss between standard multi-head attention and GQA.

<img width="900" height="250" src="https://github.com/JavierAM01/Generative-Models-of-Text/blob/main/images/GQA_arch_diagram.png">

## Training and Experimentation
- Implemented and tested various attention mechanisms.
- Trained models using Shakespeare’s works as a dataset.
- Logged results with Weights & Biases (wandb) for analysis.

## How to Run the Code
1. Train the language model:
   ```sh
   python chargpt.py --trainer.max_iters=600 --model.rope=True
   ```
2. Run unit tests for verification:
   ```sh
   python test_model.py
   ```
3. View experiment logs with Weights & Biases.

## Key Learnings
- RNNs struggle with long-term dependencies; Transformers improve contextual modeling.
- RoPE enhances positional encoding in attention layers.
- Sliding Window Attention and GQA improve efficiency without major performance losses.

<img width="900" height="500" src="https://github.com/JavierAM01/Generative-Models-of-Text/blob/main/images/ex57.png">

## Acknowledgments
This project is part of **10-623 Generative AI** at **Carnegie Mellon University**, with datasets and starter code provided by the course instructors.

