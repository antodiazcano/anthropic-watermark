# Text watermarking with a local language model

This repository is a small educational simulation of two text-watermarking
ideas:

- sampling with a secret-derived seed;
- three-layer tournament sampling.

The generator is
[`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
a 135-million-parameter causal language model. It runs locally on the CPU, so
generation and detection need no API key. The model is downloaded once on the
first run (about 260 MB).

This is not Claude's production watermark. It is intentionally short code that
shows the main generation and scoring steps.

## Language-model loop

A causal language model predicts the next token from the tokens on its left.
For every new token, this implementation does the following:

1. Run SmolLM2 on the prompt and all tokens generated so far.
2. Keep the 50 tokens with the largest logits.
3. Apply softmax to obtain a probability distribution over those tokens.
4. Measure the normalized entropy of that distribution.
5. Use the chosen watermark sampler when entropy is at least `0.5`; otherwise,
   sample normally with `np.random.choice`.
6. Append the selected token to the context and repeat.

For the non-zero probabilities $p_1,\ldots,p_k$, normalized entropy is

$$
H_{normalized} =
\frac{-\sum_{i=1}^{k}p_i\log_2(p_i)}{\log_2(k)}.
$$

It is close to `0` when one token dominates and close to `1` when many tokens
are similarly likely. Watermarking only high-entropy positions gives the
sampler room to choose among plausible alternatives.

## Seed watermark

At every selected position, the seed method:

1. computes HMAC-SHA256 with the secret as both key and message;
2. reads the first four bytes as a 32-bit integer;
3. resets NumPy to that seed;
4. samples from the model's current probability distribution.

The seed is the same at every position, but the distribution changes because
the model receives a longer context after each generated token. The selected
token can therefore change too.

For detection, the model receives the same prompt and replays the observed
tokens from left to right. At every high-entropy position, the detector
recreates the secret-seeded choice and checks whether it equals the observed
token. Its score is the fraction of matches:

$$
S_{seed} = \frac{\text{matching selected tokens}}
                 {\text{selected tokens}}.
$$

### Simple seed example

Imagine that the prompt is “The weather will”. At the first selected position,
the model gives most of its probability to `improve` and `change`. The secret
seed makes NumPy choose `change`.

The context is now “The weather will change”, so the next distribution is
different. This time the likely choices are `tomorrow` and `slowly`. NumPy is
reset to the same secret seed, but sampling from this new distribution chooses
`tomorrow`.

The watermarked continuation is therefore “change tomorrow”. During detection,
the same prompt and secret reproduce both choices, so the score is `2 / 2 = 1`.
If the second token is edited and no longer matches, the score becomes
`1 / 2 = 0.5`.

## Tournament watermark

The tournament uses three deterministic functions. Each maps the secret and a
candidate token ID to `0` or `1`, using HMAC-SHA256, SHA3-256, or keyed BLAKE2b.
Together they give every candidate a three-bit signature.

At every selected position, the method:

1. samples $2^3=8$ candidates from the current model distribution;
2. pairs them in their sampled order;
3. compares the first signature bit and keeps `1` over `0`;
4. compares the four winners using the second bit;
5. compares the two remaining winners using the third bit.

Equal bits are resolved randomly. The final winner becomes the next token.

```text
8 candidates -- bit 1 --> 4 -- bit 2 --> 2 -- bit 3 --> 1 token
```

Detection replays the model to find the same high-entropy positions, calculates
the three secret bits of each observed token, and returns their mean:

$$
S_{tournament} = \frac{\text{number of 1 bits}}
                       {3 \times \text{selected tokens}}.
$$

Ordinary text should average around `0.5` over enough tokens. Tournament
sampling favors `1` bits, so watermarked text should generally score higher.
This is statistical: a very short watermarked sample can still score lower.

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

The first-bit matches leave B, C, F, and H. In the second round, assume the tie
between B and C is resolved in favor of C, while F beats H. In the last round,
C's third bit is `1` and F's is `0`, so C wins with signature `101`.

That token contributes `(1 + 0 + 1) / 3 = 0.67` to the detection score. The
detector repeats this calculation across all selected tokens and averages the
bits.

## Running the demonstration

Open `src/main.py` and edit `secret`, `prompt`, or `length`. Running that file
from an editor generates an ordinary and a watermarked continuation for both
methods, then prints their scores. There are no command-line arguments and no
external service to configure.

Keep the generated token IDs for detection. Decoding a continuation and then
tokenizing it separately can change token boundaries, whereas passing the
original IDs exactly reproduces the generation context.

```text
src/
  base_model.py             # SmolLM2 loading, forward pass, entropy, generation
  seed_watermark.py         # secret-seeded sampling and match score
  tournament_watermark.py   # tournament sampling and mean-bit score
  main.py                   # editable demonstration
```

## Simplifications

Anthropic says Claude uses a version of SynthID-Text. The published method
derives position-specific randomness from the key and recent tokens; its
detector can score text without rerunning the generating model. This repository
instead keeps the original educational choices:

- the seed method derives one fixed seed from the secret;
- tournament bit functions use the secret and candidate token ID only;
- detection reruns SmolLM2 to recover the entropy-selected positions;
- the tournament has three layers and returns only a score, with no threshold
  or probability value.

These differences keep the complete example small, but they mean its scores are
only meaningful for this implementation, model, tokenizer, prompt, and secret.

References:

- [Anthropic: How Claude's text watermark works](https://www.anthropic.com/news/claude-text-watermark)
- [SynthID-Text paper](https://www.nature.com/articles/s41586-024-08025-4)
- [SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
