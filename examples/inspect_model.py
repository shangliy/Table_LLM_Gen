from transformers import AutoModelForCausalLM

# Load the model to inspect its architecture
model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct")

print(f"Model type: {model.config.model_type}")
print(f"\nModel architecture:")
print(model)

print(f"\n\nNamed modules (first 50):")
for i, (name, module) in enumerate(model.named_modules()):
    if i < 50:
        print(f"{name}: {type(module).__name__}")
