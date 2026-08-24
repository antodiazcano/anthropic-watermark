# Seed and tournament text watermarks

This repository contains two small educational watermarking methods built on a toy language model:

- Deterministic sampling from a secret-derived seed.
- Three-layer tournament sampling.

## Toy model and context

The model has 15 tokens. Token `i` receives probability

$$
p_i = \frac{i}{\sum_{j=0}^{14}j} = \frac{i}{105}.
$$


## Seed watermark

For every token, the seed method:

1. Obtains the model probabilities.

2. Encodes the current context as comma-separated token IDs.

3. Calculates hash based on current `context` and `secret`.

4. Converts the first four bytes to a 32-bit integer seed.

5. Seeds NumPy and samples from the unchanged probabilities.


The same query, secret, and settings reproduce the same sequence. Detection regenerates that sequence and returns its fraction of positional matches:

$$
S_{seed} = \frac{\text{matching tokens}}{\text{tokens}}.
$$

## Tournament watermark

The tournament method uses three keyed bit functions. Each function assigns `0` or `1` to a candidate using the secret, current context, and candidate token.

For each generated token, it:

1. Samples $2^3 = 8$ candidates from the model probabilities.

2. Calculates three signature bits for every candidate.

3. Pairs the candidates and lets `1` beat `0` using the first bit. If two opponents have the same bit, the winner is selected randomly.

4. Repeats with the second and third bits until one candidate remains.

```text
8 candidates -- bit 1 --> 4 -- bit 2 --> 2 -- bit 3 --> 1 token
```

Detection does not regenerate the tournament. For every received token, it
recreates the three bits from that token and its context, then returns their
mean:

$$
S_{tournament} = \frac{1}{3T}
    \sum_{t=1}^{T}\sum_{j=1}^{3}g_j(c_t, x_t).
$$

Text sampled independently of the secret should score around `0.5` over enough
tokens. Tournament sampling favors `1` bits, so matching watermarked text should
usually score higher.

## Simple examples

### 1. Seed method

Use the query `The garden hidden`, the secret `hello`, and a context length of `3`.

1. The query becomes the context `[0, 5, 10]`.

2. The model produces its 15 token probabilities.

3. HMAC hashes the secret and the context string `0,5,10`.

4. The first four hash bytes produce the seed `3470980064`.

5. NumPy samples token `7`, which represents the word `shapes`.

6. The new context is `[5, 10, 7]`, and the process repeats with a new hash and seed.

After three steps, the generated sequence is:

```text
[7, 14, 13] -> shapes . again
```

For detection, the model regenerates `[7, 14, 13]`. If the received sequence is
`[7, 14, 0]`, only the first two positions match, so the score is:

$$
S_{seed} = \frac{2}{3} = 0.67.
$$

### 2. Tournament method

Use the same query, secret, and context. For this reproducible example, ordinary
NumPy sampling starts from seed `7`. This seed only fixes the example's random
candidate draws; it is not derived from the watermark secret.

The model samples eight candidates and calculates their three signature bits:

| Candidate | Token | Word | Signature |
|---:|---:|---|---|
| 1 | 4 | `river` | `[0, 0, 1]` |
| 2 | 13 | `again` | `[0, 0, 0]` |
| 3 | 10 | `hidden` | `[0, 1, 1]` |
| 4 | 12 | `today` | `[0, 0, 0]` |
| 5 | 14 | `.` | `[1, 0, 1]` |
| 6 | 11 | `distant` | `[1, 1, 1]` |
| 7 | 10 | `hidden` | `[0, 1, 1]` |
| 8 | 4 | `river` | `[0, 0, 1]` |

The tournament then reduces the candidates:

1. Using the first bit, the winners are `[4, 10, 11, 10]`.

2. Using the second bit, the winners are `[10, 11]`.

3. Using the third bit, the final winner is token `10`, which is the word `hidden`.

To generate another token, `10` is appended to the context, making the new three-token context `[5, 10, 10]`, and a new eight-candidate tournament begins.

For detection, only the received token and its context are needed. The detector recreates token `10`'s signature as `[0, 1, 1]`, so this token contributes:

$$
S_{tournament} = \frac{0 + 1 + 1}{3} = 0.67.
$$

For a longer text, the detector repeats this calculation for every received token and averages all the bits.
