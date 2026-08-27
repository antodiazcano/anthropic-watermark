"""Compare ordinary and watermarked generation with a local language model."""

from time import time

import numpy as np
from dotenv import load_dotenv
from transformers.utils import logging as transformers_logging

from src.base_model import LanguageModel
from src.seed_watermark import SeedWatermarkModel
from src.tournament_watermark import TournamentWatermarkModel


def check(model: LanguageModel, prompt: str, length: int) -> None:
    """Generate and score an ordinary and a watermarked continuation."""

    seed = 9
    prompt_tokens = model.text_to_tokens(prompt)

    np.random.seed(seed)
    plain_tokens = prompt_tokens + model.generate(prompt, length, watermark=False)
    plain_text = model.tokens_to_text(plain_tokens)

    np.random.seed(seed)
    watermarked_tokens = prompt_tokens + model.generate(prompt, length, watermark=True)
    watermarked_text = model.tokens_to_text(watermarked_tokens)

    plain_score = model.detect_tokens(plain_tokens)
    watermarked_score = model.detect_tokens(watermarked_tokens)

    print(f"Plain text: {plain_text}")
    print(f"\nWatermarked text: {watermarked_text}")
    print(f"\nPlain score: {plain_score:.2f}")
    print(f"\nWatermarked score: {watermarked_score:.2f}")


def main() -> None:
    """Run both watermark demonstrations."""

    # Load HF token and disable progress bar print
    load_dotenv()
    transformers_logging.disable_progress_bar()

    secret = "hello"
    prompt = "Artificial intelligence will"
    length = 30

    t = time()
    print("SEED WATERMARK\n")
    check(SeedWatermarkModel(secret), prompt, length)
    print(f"\nElapsed time: {time() - t:.2f} seconds.")

    t = time()
    print("\n\nTOURNAMENT WATERMARK\n")
    check(TournamentWatermarkModel(secret), prompt, length)
    print(f"\nElapsed time: {time() - t:.2f} seconds.")


if __name__ == "__main__":
    main()
