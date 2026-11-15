"""Convenience entry point for the Milling Stage 2 pipeline."""
from src.milling_pipeline import parse_args, run


if __name__ == "__main__":
    run(parse_args())
