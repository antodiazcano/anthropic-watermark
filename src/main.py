"""Script to compare normal sampling with watermark sampling."""

from src.base_model import DummyLLM
from src.seed_watermark import SeedWatermarkModel
from src.tournament_watermark import TournamentWatermarkModel


def check(model: DummyLLM) -> None:
    """Checks if the watermarks are implemented correctly.

    Args:
        model: Model used to implement the watermark.
    """

    text_length = 10

    plain_tokens = model.generate(text_length, watermark=False)
    watermarked_tokens = model.generate(text_length, watermark=True)

    plain_score = model.detect(plain_tokens)
    watermarked_score = model.detect(watermarked_tokens)

    print(f"Plain text: {model.tokens_to_text(plain_tokens)}")
    print(f"\nWatermarked text: {model.tokens_to_text(watermarked_tokens)}")

    print(f"\nScore plain: {plain_score:.2f}")
    print(f"\nScore watermarked: {watermarked_score:.2f}")


def main() -> None:
    """Checks the seed and tournament watermark models."""

    secret = "hello"

    check(SeedWatermarkModel(secret))
    print("\n" * 3)
    check(TournamentWatermarkModel(secret))


if __name__ == "__main__":
    main()
