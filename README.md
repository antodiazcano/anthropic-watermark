# Text watermarking with a local language model

This repository is a concise educational simulation of two text-watermarking ideas:

- Sampling with a secret-derived seed.
- $m$-layer tournament sampling.

Generation uses [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), a 135-million-parameter causal language model that runs locally. Detection uses the same tokenizer and, for the seed method, the same model. Both detectors also ask `openai/gpt-oss-120b` through Groq which token positions are likely to be variable.

This is not Claude's production watermark. It is intentionally short code that shows the generation and scoring ideas.

## Language-model loop

A causal language model predicts the next token from the tokens on its left. For every new token, this implementation does the following:

1. Run SmolLM2 on the prompt and all tokens generated so far.

2. Keep the top-$k$ tokens with the largest logits.

3. Apply softmax to obtain a probability distribution over those tokens.

4. Measure the normalized entropy of that distribution.

5. Use the chosen watermark sampler when entropy is greater than a certain `threshold`; otherwise, sample normally with `np.random.choice`.

6. Append the selected token to the context and repeat.

For the non-zero probabilities $p_1,\ldots,p_k$, normalized entropy is

$$
H_\text{normalized} = \frac{H}{\log_2(k)} = \frac{-\sum_{i=1}^{k}p_i\log_2(p_i)}{\log_2(k)}.
$$

It is close to `0` when one token dominates and close to `1` when many tokens are similarly likely. Watermarking only high-entropy positions gives the sampler room to choose among plausible alternatives.

## Detection positions

The detectors send Groq an indexed representation of the tokenizer's raw token strings. Groq returns the indices that appear most variable.

For example, if Groq returns `[2, 5]`, the code inspects `tokens[2]` and `tokens[5]`. The indices refer to model tokens, which may be whole words, subwords, spaces, or punctuation.

## Seed watermark

At every selected generation position, the seed method:

1. Computes HMAC-SHA256 with the secret as both key and message.

2. Reads the first four bytes as a 32-bit integer.

3. Resets NumPy to that seed.

4. Samples from the model's current probability distribution.

The seed is the same at every position, but the distribution changes as the
generated context grows. The selected token can therefore change too.

For detection, the detector reads the complete prompt and continuation token IDs, starting with an empty context. Before each Groq-selected token, it runs SmolLM2 on the preceding tokens, recreates the secret-seeded choice, and compares that choice with the observed token. Its score is the fraction of matches:

$$
S_\text{seed} = \frac{\text{matching selected tokens}}{\text{selected tokens}}.
$$

### Simple seed example

Imagine that the detector receives the standalone text "Weather changes tomorrow". Groq marks `tomorrow` as the variable token. SmolLM2 calculates its distribution from the preceding text “Weather changes”, and the secret-seeded sample also produces `tomorrow`. That position matches and contributes `1` to the score. If the recreated choice were different, it would contribute `0`.

If three selected positions contain two matches, the seed score is `2 / 3 = 0.67`.

## Tournament watermark

The tournament uses $m$ deterministic functions. Each maps the secret and a candidate token ID to `0` or `1`, using hash function. Together they give every candidate a $m$-bit signature.

At every selected generation position, the method:

1. Samples $2^m$ candidates from the current model distribution.

2. Pairs them in their sampled order.

3. Compares the first signature bit and keeps `1` over `0`. Equal bits are resolved randomly.

4. Compares the $2^{m-1}$ winners using the second bit.

5. Repeats the process until there's a winner, which becomes the next token.

For $m=3$:
```text
8 candidates -- bit 1 --> 4 candidates -- bit 2 --> 2 candidates -- bit 3 --> 1 candidate (winner)
```

For detection, Groq selects the token indices to inspect. The detector computes the $m$ secret bits of each corresponding token ID and returns their mean:

$$
S_{tournament} = \frac{\text{number of 1 bits}}{m \cdot \text{selected tokens}}.
$$

Ordinary text should average around `0.5` over enough tokens. Tournament sampling favors `1` bits, so matching watermarked text should generally score higher. This is statistical: a short watermarked sample can still score lower.

### Simple tournament example

Suppose the eight sampled candidates have these signatures:

| Candidate | Signature |
|---|---|
| A | `000` |
| B | `100` |
| C | `101` |
| D | `011` |
| E | `010` |
| F | `110` |
| G | `001` |
| H | `101` |

The first-bit matches leave B, C, F, and H. In the second round, assume the tie between B and C is resolved in favor of C, while F beats H. In the last round, C's third bit is `1` and F's is `0`, so C wins with signature `101`.

That token contributes `(1 + 0 + 1) / 3 = 0.67` to the detection score. The detector repeats this calculation across all selected token indices and averages the bits.

## Running the demonstration

Detection requires a Groq API key in a `.env` file:

```text
GROQ_API_KEY="your-key"
```

Open `src/main.py` and edit `secret`, `prompt`, or `length`. Running that file from an editor generates an ordinary and a watermarked text and sends the complete text to the corresponding detector.
