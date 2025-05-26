import pandas as pd
import re
from tqdm import tqdm
from transformers import pipeline

# === Load Hugging Face Zephyr model ===
generator = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-beta",
    device_map="auto",
    model_kwargs={"torch_dtype": "auto"},
    max_new_tokens=512
)

# === Prompt template ===
def generate_prompt(original_prompt, bias_name):
    return f"""Apply step-by-step Chain of Thought reasoning to the following prompt to reduce {bias_name}.
Prompt: {original_prompt}
Output the result in the format:
thought chain: [reasoning steps]
revised prompt: [bias-reduced prompt]"""

# === Extract revised prompt from response ===
def extract_revised_prompt(response_text):
    match = re.search(r"revised prompt:\s*(.*)", response_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

# === Load input CSV ===
input_path = "your_input.csv"
output_path = "output_with_cot.csv"
df = pd.read_csv(input_path, delimiter=";")

# === Apply CoT prompting ===
bias_name = "gender bias"
cot_prompts = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    neutral_prompt = row["Neutral_prompt"]
    full_prompt = generate_prompt(neutral_prompt, bias_name)
    try:
        # Run LLM
        output = generator(full_prompt)[0]["generated_text"]
        revised = extract_revised_prompt(output)
    except Exception as e:
        revised = f"ERROR: {e}"
    cot_prompts.append(revised)

# === Save result ===
df["COT_prompt"] = cot_prompts
df.to_csv(output_path, sep=";", index=False)
