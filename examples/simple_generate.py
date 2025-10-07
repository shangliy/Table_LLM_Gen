"""
Simple script to load a pretrained GReaT model and generate synthetic data to CSV.
"""

from be_great import GReaT
import logging
from utils import set_logging_level

logger = set_logging_level(logging.INFO)

# Configuration
MODEL_PATH = "llama_lora_iris"  # Change this to your model path
OUTPUT_CSV = "synthetic_data.csv"
N_SAMPLES = 100
TEMPERATURE = 0.7
DEVICE = "cuda"  # Use "cpu" if you don't have a GPU

# Load the pretrained model
logger.info(f"Loading model from {MODEL_PATH}...")
model = GReaT.load_from_dir(MODEL_PATH)
logger.info("Model loaded successfully!")

# Generate synthetic data
logger.info(f"Generating {N_SAMPLES} synthetic samples...")
synthetic_data = model.sample(
    n_samples=N_SAMPLES,
    temperature=TEMPERATURE,
    device=DEVICE,
)

# Save to CSV
logger.info(f"Saving to {OUTPUT_CSV}...")
synthetic_data.to_csv(OUTPUT_CSV, index=False)

# Display results
logger.info(f"Successfully generated {len(synthetic_data)} samples!")
print("\nFirst 10 samples:")
print(synthetic_data.head(10))

print("\nStatistics:")
print(synthetic_data.describe())
