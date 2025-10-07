from be_great import GReaT
from sklearn import datasets
import logging
from utils import set_logging_level

logger = set_logging_level(logging.INFO)

# Load sample data
data = datasets.load_iris(as_frame=True).frame
print(data.head())

column_names = ["sepal length", "sepal width", "petal length", "petal width", "target"]
data.columns = column_names

# Initialize GReaT with LLaMA and LoRA
# Note: You can use different LLaMA models like:
# - "meta-llama/Llama-2-7b-hf"
# - "meta-llama/Llama-3.2-1B"
# - "meta-llama/Meta-Llama-3-8B"
great = GReaT(
    ##llm="unsloth/Llama-3.2-1B-Instruct",  # LLaMA model from HuggingFace
    llm="unsloth/gemma-3-270m-it",  # LLaMA model from HuggingFace
    efficient_finetuning="lora",  # Enable LoRA fine-tuning
    epochs=50,
    batch_size=1,  # Smaller batch size to reduce memory usage
    save_steps=100,
    logging_steps=5,
    experiment_dir="trainer_llama_lora",
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
)

# Train the model
trainer = great.fit(data, column_names=column_names)

# Save the trained model
great.save("llama_lora_iris")

# Generate synthetic samples
print("\nGenerating synthetic samples...")
synthetic_data = great.sample(n_samples=10, temperature=0.7, device="cuda")
print(synthetic_data)
