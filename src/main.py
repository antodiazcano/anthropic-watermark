"""Compare ordinary and watermarked generation with a local language model."""

import numpy as np

from src.base_model import LanguageModel
from src.seed_watermark import SeedWatermarkModel
from src.tournament_watermark import TournamentWatermarkModel


def check(model: LanguageModel, prompt: str, length: int) -> None:
    """Generate and score an ordinary and a watermarked continuation."""

    np.random.seed(7)
    plain_tokens = model.generate(prompt, length, watermark=False)

    np.random.seed(7)
    watermarked_tokens = model.generate(prompt, length, watermark=True)

    plain_score = model.detect(prompt, plain_tokens)
    watermarked_score = model.detect(prompt, watermarked_tokens)

    print(f"Prompt: {prompt}")
    print(f"Plain continuation: {model.tokens_to_text(plain_tokens)}")
    print(f"Watermarked continuation: {model.tokens_to_text(watermarked_tokens)}")
    print(f"Plain score: {plain_score:.2f}")
    print(f"Watermarked score: {watermarked_score:.2f}")


def main() -> None:
    """Run both watermark demonstrations."""

    secret = "hello"
    prompt = "Artificial intelligence will"
    length = 30

    print("SEED WATERMARK\n")
    check(SeedWatermarkModel(secret), prompt, length)

    print("\nTOURNAMENT WATERMARK\n")
    check(TournamentWatermarkModel(secret), prompt, length)


if __name__ == "__main__":
    main()
