"""Compare ordinary and watermarked generation with a local language model."""

from time import time

import numpy as np

from src.base_model import LanguageModel
from src.seed_watermark import SeedWatermarkModel
from src.tournament_watermark import TournamentWatermarkModel


def check(model: LanguageModel, prompt: str, length: int) -> None:
    """Generate and score an ordinary and a watermarked continuation."""

    seed = 9

    np.random.seed(seed)
    plain_tokens = model.generate(prompt, length, watermark=False)
    plaint_text = prompt + model.tokens_to_text(plain_tokens)

    np.random.seed(seed)
    watermarked_tokens = model.generate(prompt, length, watermark=True)
    watermarked_text = prompt + model.tokens_to_text(watermarked_tokens)

    plain_score = model.detect(plaint_text)
    watermarked_score = model.detect(watermarked_text)

    print(f"Plain text: {plaint_text}")
    print(f"Watermarked text: {watermarked_text}")
    print(f"Plain score: {plain_score:.2f}")
    print(f"Watermarked score: {watermarked_score:.2f}")


def main() -> None:
    """Run both watermark demonstrations."""

    secret = "hello"
    prompt = "Artificial intelligence will "
    length = 30

    t = time()
    print("SEED WATERMARK\n")
    check(SeedWatermarkModel(secret), prompt, length)
    print(f"Elapsed time: {time() - t:.2f} seconds.")

    t = time()
    print("\n\nTOURNAMENT WATERMARK\n")
    check(TournamentWatermarkModel(secret), prompt, length)
    print(f"Elapsed time: {time() - t:.2f} seconds.")


if __name__ == "__main__":
    main()
