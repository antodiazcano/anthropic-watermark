# Seed and tournament text watermarks

This repository contains two educational watermarking methods built on a small
toy language model:

- sampling with a secret-derived seed;
- three-layer tournament sampling.

## Toy model

The vocabulary contains 15 tokens. Token `i`, for `i` from `0` to `14`, has
probability

$$
p_i = \frac{i+1}{\sum_{j=1}^{15}j} = \frac{i+1}{120}.
$$

The toy model returns this same distribution at every position. A real language
model would instead produce a new distribution from the preceding text.

Before watermarking a position, the generator calculates normalized entropy:

$$
H_{normalized} =
\frac{-\sum_{i:p_i>0}p_i\log_2(p_i)}{\log_2(V)}.
$$

The position is watermarked when this value is at least `0.5`. Otherwise, it is
sampled normally. The toy distribution has normalized entropy `0.9395`, so all
of its positions are watermarked when watermarking is enabled.

## Seed watermark

For every selected position, the seed method:

1. obtains the model probabilities;
2. calculates `HMAC-SHA256` using the secret as both key and message;
3. converts the first four digest bytes to a big-endian 32-bit seed;
4. seeds NumPy;
5. samples from the unchanged model probabilities.

Detection first asks a language model to identify variable token positions. For
each selected position, it recreates the seeded choice and compares that choice
with the received token. If $I$ is the set of selected indices, the score is

$$
S_{seed} = \frac{1}{|I|}\sum_{t\in I}
    \mathbb{1}[x_t=\hat{x}_t].
$$

Here, $x_t$ is the received token and $\hat{x}_t$ is the token reproduced from
the secret and model probabilities.

### Simple seed example

Use the secret `hello` and the toy probabilities:

1. Their normalized entropy is `0.9395`, so the position is selected.
2. HMAC produces the seed `2993689727`.
3. Seeded sampling selects token `11`, which represents `distant`.
4. The toy distribution is unchanged at the next position, so the same process
   selects token `11` again.

After three positions, the watermarked tokens are

```text
[11, 11, 11] -> distant distant distant
```

Assume detection inspects all three positions but receives `[11, 11, 0]`. It
recreates token `11` at every position and finds two matches, giving

$$
S_{seed} = \frac{2}{3} = 0.67.
$$

With a real model, the probabilities can change at every position, so the same
seed does not necessarily select the same token.

## Tournament watermark

The tournament method currently uses three deterministic bit functions. Each
function assigns `0` or `1` from the secret and candidate token ID. The current
functions use HMAC-SHA256, SHA3-256, and keyed BLAKE2b.

For every selected position, the method:

1. samples $2^3=8$ candidates from the model probabilities;
2. calculates three signature bits for each candidate;
3. pairs candidates and compares their first bits;
4. lets `1` beat `0`, selecting randomly when the bits match;
5. repeats with the second and third bits until one candidate remains.

```text
8 candidates -- bit 1 --> 4 -- bit 2 --> 2 -- bit 3 --> 1 token
```

Tournament detection does not recreate candidate draws or run the tournament.
It calculates the three bits of each selected received token and returns their
mean. For $m=3$ functions and selected index set $I$,

$$
S_{tournament} = \frac{1}{3|I|}
    \sum_{t\in I}\sum_{j=1}^{3}g_j(x_t).
$$

Plain text should score around `0.5` over enough tokens. Tournament sampling
favors `1` bits, so matching watermarked text should generally score higher.

### Simple tournament example

Use the secret `hello` and the toy probabilities. To make the random candidate
draws reproducible for this example, assume ordinary NumPy sampling starts from
seed `7`.

The eight candidates and their signatures are

| Candidate | Token | Word | Signature |
|---:|---:|---|---|
| 1 | 3 | `quiet` | `[0, 1, 1]` |
| 2 | 13 | `again` | `[0, 1, 1]` |
| 3 | 9 | `new` | `[1, 0, 0]` |
| 4 | 12 | `today` | `[1, 0, 1]` |
| 5 | 14 | `.` | `[1, 0, 1]` |
| 6 | 10 | `hidden` | `[0, 1, 0]` |
| 7 | 10 | `hidden` | `[0, 1, 0]` |
| 8 | 3 | `quiet` | `[0, 1, 1]` |

The three rounds produce

1. first-bit winners: `[3, 9, 14, 3]`;
2. second-bit winners: `[3, 3]`;
3. third-bit winner: token `3`, or `quiet`.

The detector recreates token `3`'s signature as `[0, 1, 1]`, so this token
contributes

$$
\frac{0+1+1}{3} = 0.67
$$

to the overall tournament score. For longer text, detection repeats this for
every selected position and averages all signature bits.

## Detection model

Both detectors use `openai/gpt-oss-120b` through Groq to choose token positions
that appear variable. Detection therefore requires `GROQ_API_KEY` in a `.env`
file. Generation itself does not call Groq.

## Code

```text
src/
  base_model.py             # toy model, entropy selection, and detection selection
  seed_watermark.py         # seeded sampling and match score
  tournament_watermark.py   # tournament sampling and mean-bit score
  main.py                   # editable demonstration
```

This is a small educational model, not a detector for text produced by real
language models.
