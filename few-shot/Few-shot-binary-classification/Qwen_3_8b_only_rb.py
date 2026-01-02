import os
import pandas as pd
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import sys

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 16
TARGET_GPU = "" # Target GPU node
MODEL_ID = "Qwen/Qwen3-8B" # Model ID
BASE_DATA_PATH = "" # Base Data Path
OUTPUT_DIR = "" # Output Directory

# Only Regional Bias Path 
REGIONAL_BIAS_EXAMPLES_PATH = os.path.join(BASE_DATA_PATH, "")

# Inference File
INFERENCE_FILENAME = "" 
INFERENCE_FILE_PATH = os.path.join(BASE_DATA_PATH, INFERENCE_FILENAME)

# Saving the results
RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "classification_results_final.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "classification_report.txt")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

MODEL_PROMPT = """You are an expert in identifying regional biases in social media comments about Indian states and regions. Your task is to classify whether a comment contains regional biases or not.

Task: Classify the given comment as either "REGIONAL BIAS" (1) or "NON-REGIONAL BIAS" (0).

Definition:
- REGIONAL BIAS (1): Comments that reinforce or propagate biases about Indian states, regions, or their people based on linguistic, cultural, economic, political, or infrastructural aspects. The comments can reflect either positive or negative biases towards specific states or regions.
- NON-REGIONAL BIAS (0): Comments that are neutral or factual without generalisations, or unrelated to regional characteristics.

Step-by-Step Analysis Process:
Step 1: Identify Regional References
Think: Does this comment mention or refer to:
- Specific Indian states (e.g., Bihar, Kerala, Punjab, etc.)
- Regional groups (e.g., South Indians, North Indians, Biharis, etc.)
- Cities or regions within India
- Language communities within India

Step 2: Check for Elements reinforcing biases
Look for these patterns:
- Generalisations about people from a state or a regional group ("All X are Y")
- Assumptions about state/regional characteristics
- Comparative statements that imply superiority/inferiority
- Overgeneralized cultural, linguistic, economic, political, or infrastructural claims

Step 3: Assess the Nature of the Statement
Consider:
- Is this a factual observation or a generalised assumption?
- Does it reinforce existing biases?
- Is it based on a broad generalisation?
- Does it perpetuate divisions?

Step 4: Final Classification
Based on the analysis above, classify as:
- REGIONAL BIAS (1) if the comment reinforces regional biases or stereotypes
- NON-REGIONAL BIAS (0) if the comment is neutral, factual, or doesn't contain regional bias.

Your response must include a brief line of reasoning followed by the final classification in the format "Classification: [0 or 1]"."""


def get_device(target_device: str):
    # Sets up GPU visibility.
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if "cuda" in target_device:
        try:
            device_idx = int(target_device.split(":")[-1])
            if device_idx >= torch.cuda.device_count():
                print(f"Device {target_device} not found. Switching to cuda:0", flush=True)
                return torch.device("cuda:0")
        except ValueError:
            pass 
    return torch.device(target_device)

def load_model_and_tokenizer(model_id: str, device: str):
    # Handles authentication and loads the full model.
    print(f"Loading model: {model_id} in 16-bit precision on {device}.", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16, 
            device_map=device,
            trust_remote_code=True
        )
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        print(f" Error loading model: {e}")
        sys.exit(1)

def load_few_shot_examples(regional_path: str, total_examples: int) -> tuple[str, set]:
    # Loads ONLY regional bias examples.
    print(f"Loading {total_examples} regional bias examples...", flush=True)
    all_examples = []

    if not os.path.exists(regional_path):
        print(f" FATAL ERROR: File not found: {regional_path}")
        sys.exit(1)
    
    df = pd.read_csv(regional_path)
    target_col = 'level-1' if 'level-1' in df.columns else 'label'

    filtered = df[df[target_col] == 1]
    
    if len(filtered) < total_examples:
        print(f" Warning: Requested {total_examples} examples but file only has {len(filtered)}. Using all available.")
        n = len(filtered)
    else:
        n = total_examples
    
    reg_samples = filtered.sample(n=n, random_state=SEED)
    for _, row in reg_samples.iterrows():
        all_examples.append({'comment': str(row['comment']), 'label': 1})

    random.shuffle(all_examples)

    examples_str = ""
    used_comments_set = set()
    for example in all_examples:
        comment = example['comment'].strip()
        label = example['label']
        used_comments_set.add(comment)
        reasoning = "Contains specific regional stereotyping/generalization."
        examples_str += f"\n--- Example ---\nComment: \"{comment}\"\n<think>\n{reasoning}\n</think>\nClassification: {label}\n--- End Example ---\n"

    print(f"   Loaded {len(all_examples)} examples total.", flush=True)
    return examples_str, used_comments_set

def parse_response(response: str) -> tuple[str, int]:
    # Robustly parses a single model response text to ensure a 0 or 1 output.
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    reasoning = think_match.group(1).strip() if think_match else "No reasoning provided"

    if re.search(r"Classification:?\s*1", response, re.IGNORECASE):
        return reasoning, 1
    if re.search(r"Classification:?\s*0", response, re.IGNORECASE):
        return reasoning, 0

    return reasoning, 0 

def main():
    # Main function to orchestrate the entire classification process.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device_obj = get_device(TARGET_GPU)
    device_str = str(device_obj)
    if "cuda" in device_str and ":" not in device_str:
        device_str = "cuda:0"

    print(f"Using device: {device_str}", flush=True)

    # Load 50 examples
    few_shot_prompt, used_comments = load_few_shot_examples(
        regional_path=REGIONAL_BIAS_EXAMPLES_PATH,
        total_examples=50 
    )
    
    print(f"\nLoading main dataset from: {INFERENCE_FILE_PATH}", flush=True)
    if not os.path.exists(INFERENCE_FILE_PATH):
        print(f"FATAL ERROR: Inference file not found at {INFERENCE_FILE_PATH}")
        sys.exit(1)
        
    df_full = pd.read_csv(INFERENCE_FILE_PATH)
    print(f"Initial dataset size: {len(df_full)}")

    print(f"Filtering out {len(used_comments)} few-shot examples.", flush=True)
    df = df_full[~df_full['comment'].astype(str).str.strip().isin(used_comments)].copy()
    print(f"Final inference dataset size: {len(df)} (Should be ~24950 if original was 25000)", flush=True)

    processed_count = 0
    if os.path.exists(RESULTS_CSV_PATH):
        try:
            existing_results = pd.read_csv(RESULTS_CSV_PATH)
            processed_count = len(existing_results)
            print(f"Resuming: Found {processed_count} completed comments.", flush=True)
        except pd.errors.EmptyDataError:
            processed_count = 0

    if processed_count >= len(df):
        print("All comments already processed!", flush=True)
        df_to_process = pd.DataFrame()
    else:
        df_to_process = df.iloc[processed_count:]
        print(f"--> Processing remaining {len(df_to_process)} comments.", flush=True)

    # Model Loading & Inference
    if not df_to_process.empty:
        model, tokenizer = load_model_and_tokenizer(MODEL_ID, device=device_str)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(df_to_process), BATCH_SIZE), desc="Inference"):
                batch_df = df_to_process.iloc[i:i+BATCH_SIZE]
                batch_prompts = []
                
                for _, row in batch_df.iterrows():
                    comment_text = str(row['comment'])
                    user_msg = (f"{few_shot_prompt}\n"
                                f"--- Classify the following comment ---\n"
                                f"Comment: \"{comment_text}\"")
                    
                    messages = [
                        {"role": "system", "content": MODEL_PROMPT},
                        {"role": "user", "content": user_msg}
                    ]
                    
                    prompt_str = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    batch_prompts.append(prompt_str)

                inputs = tokenizer(
                    batch_prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=4096 
                ).to(model.device)

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False, 
                    temperature=0.0
                )
                
                decoded_responses = tokenizer.batch_decode(
                    generated_ids[:, inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
                
                batch_results = []
                for idx, raw_response in enumerate(decoded_responses):
                    reasoning, prediction = parse_response(raw_response)
                    original_row = batch_df.iloc[idx]
                    
                    batch_results.append({
                        'comment': original_row['comment'],
                        'true_label': original_row['level-1'], 
                        'predicted_label': prediction,
                        'model_response': reasoning
                    })

                res_df = pd.DataFrame(batch_results)
                is_first_write = not os.path.exists(RESULTS_CSV_PATH)
                res_df.to_csv(RESULTS_CSV_PATH, mode='a', header=is_first_write, index=False)

    print(f"\nProcessing Complete. Results saved to {RESULTS_CSV_PATH}")

    # Generate Report
    if os.path.exists(RESULTS_CSV_PATH):
        print("Generating Classification Report.", flush=True)
        full_df = pd.read_csv(RESULTS_CSV_PATH)
        
        y_true = pd.to_numeric(full_df['true_label'], errors='coerce').fillna(0).astype(int)
        y_pred = pd.to_numeric(full_df['predicted_label'], errors='coerce').fillna(0).astype(int)
        
        report = classification_report(y_true, y_pred, target_names=['Non-Biased (0)', 'Biased (1)'])
        print(report)
        
        with open(REPORT_PATH, 'w') as f:
            f.write(report)
            
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Biased', 'Biased'],
                    yticklabels=['Non-Biased', 'Biased'])
        plt.title('Confusion Matrix: Qwen 8B Regional Bias')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(CONFUSION_MATRIX_PATH)
        print(f"Confusion Matrix saved to {CONFUSION_MATRIX_PATH}")

if __name__ == "__main__":
    main()
