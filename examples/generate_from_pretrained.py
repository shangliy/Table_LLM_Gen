"""
Script to load a pretrained GReaT model and generate synthetic data to CSV.

Usage:
    python generate_from_pretrained.py --model_path <path_to_model> --output <output.csv> --n_samples 100

Example:
    python generate_from_pretrained.py --model_path llama_lora_iris --output synthetic_data.csv --n_samples 1000
"""

import argparse
import logging
from be_great import GReaT
from utils import set_logging_level

# Set up logging
logger = set_logging_level(logging.INFO)


def generate_synthetic_data(
    model_path: str,
    output_path: str,
    n_samples: int = 100,
    temperature: float = 0.7,
    device: str = "cuda",
    guided_sampling: bool = False,
    random_feature_order: bool = True,
):
    """
    Load a pretrained GReaT model and generate synthetic data.

    Args:
        model_path: Path to the saved GReaT model directory
        output_path: Path where to save the generated CSV file
        n_samples: Number of synthetic samples to generate
        temperature: Sampling temperature (lower = more conservative, higher = more diverse)
        device: Device to use for generation ('cuda' or 'cpu')
        guided_sampling: Whether to use guided feature-by-feature sampling
        random_feature_order: Whether to randomize feature order in guided sampling
    """
    # Load the pretrained model
    logger.info(f"Loading pretrained model from: {model_path}")
    model = GReaT.load_from_dir(model_path)
    logger.info("Model loaded successfully!")

    # Display model information
    logger.info(f"Model features: {model.columns}")
    logger.info(f"Numerical columns: {model.num_cols}")
    logger.info(f"Conditional column: {model.conditional_col}")

    # Generate synthetic data
    logger.info(f"Generating {n_samples} synthetic samples...")
    logger.info(f"Parameters: temperature={temperature}, device={device}, guided_sampling={guided_sampling}")

    synthetic_data = model.sample(
        n_samples=n_samples,
        temperature=temperature,
        device=device,
        guided_sampling=guided_sampling,
        random_feature_order=random_feature_order,
    )

    # Save to CSV
    logger.info(f"Saving synthetic data to: {output_path}")
    synthetic_data.to_csv(output_path, index=False)
    logger.info(f"Successfully saved {len(synthetic_data)} samples to {output_path}")

    # Display sample of generated data
    logger.info("\nSample of generated data:")
    print(synthetic_data.head(10))

    # Display basic statistics
    logger.info("\nBasic statistics of generated data:")
    print(synthetic_data.describe())

    return synthetic_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data from a pretrained GReaT model"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved GReaT model directory",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_data.csv",
        help="Output CSV file path (default: synthetic_data.csv)",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of synthetic samples to generate (default: 100)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature, controls diversity (default: 0.7)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for generation (default: cuda)",
    )

    parser.add_argument(
        "--guided_sampling",
        action="store_true",
        help="Enable guided feature-by-feature sampling (slower but more reliable)",
    )

    parser.add_argument(
        "--no_random_order",
        action="store_true",
        help="Disable random feature order in guided sampling",
    )

    args = parser.parse_args()

    # Generate synthetic data
    generate_synthetic_data(
        model_path=args.model_path,
        output_path=args.output,
        n_samples=args.n_samples,
        temperature=args.temperature,
        device=args.device,
        guided_sampling=args.guided_sampling,
        random_feature_order=not args.no_random_order,
    )


if __name__ == "__main__":
    main()
