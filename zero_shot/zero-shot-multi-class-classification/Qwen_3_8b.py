import os
import re
import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from huggingface_hub import login
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_ID = "Qwen/Qwen3-8B" 
BATCH_SIZE = 16

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Warning: HF_TOKEN not found in environment variables. If your model is private/gated, this might fail.")
    HF_TOKEN = "your_actual_token_here" # Hugging_face Token
  
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("Successfully logged into Hugging Face.")
    except Exception as e:
        print(f"Login failed: {e}")

def get_classifier_prompt(): """

You are an expert content moderator. Your task is to analyze the following comment for regional bias and assign a severity score (1, 2, or 3).

**Instructions:**
1. Analyze the comment for stereotypes, hostility, or hateful language.
2. Determine the severity based on these definitions:
   * **3 (Severe):** Overtly offensive, hateful, slurs, or exclusionary language.
   * **2 (Moderate):** Clear negative generalizations or explicit stereotypes without slur usage.
   * **1 (Mild):** Subtle bias, positive stereotypes, or non-offensive remarks.

**Comment:**
"{comment_text}"

**Reasoning:**
(Provide a brief thought process)

**Final Classification:**
"""

def parse_model_output(generated_text):
    """
    Extracts the integer class (1, 2, or 3) from the model's response.
    Returns 1 (Mild) as a fallback if the model hallucinates.
    """
    try:
        if "Final Classification:" in generated_text:
            answer_part = generated_text.split("Final Classification:")[-1]
            match = re.search(r'\d', answer_part)
            if match:
                val = int(match.group(0))
                if val in [1, 2, 3]:
                    return val
    except Exception:
        pass
    
    # Fallback default
    return 1

def load_inference_stack(gpu_id):
    """
    Loads the Model and Tokenizer.
    """
    device_str = f"cuda:{gpu_id}"
    print(f"Loading {MODEL_ID} onto {device_str}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16, 
            device_map=device_str,
            trust_remote_code=True
        )
        print("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Critical Error loading model: {e}")
        exit(1)


def main(args):
    if not torch.cuda.is_available():
        print("Error: CUDA is missing. You really need a GPU for this.")
        return
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Load Data
    print(f"Reading input file: {args.input_csv}")
    try:
        df_full = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print("Error: Input CSV not found.")
        return

    if 'comment' not in df_full.columns:
        print("Error: Your CSV must have a 'comment' column.")
        return

    all_comments = df_full['comment'].astype(str).tolist()
    total_count = len(all_comments)
    results_path = os.path.join(args.output_dir, "classification_results_in_progress.csv")
    start_idx = 0
    
    if os.path.exists(results_path):
        try:
            existing_df = pd.read_csv(results_path, engine='python')
            start_idx = len(existing_df)
            print(f"Resume file found. Picking up from row {start_idx + 1}...")
        except pd.errors.EmptyDataError:
            print("Progress file exists but is empty. Starting from scratch.")
    else:
        print("Starting a fresh inference run.")

    # Initialize Pipeline
    model, tokenizer = load_inference_stack(args.gpu_id)
    
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    prompt_template = get_classifier_prompt()

    print("\nStarting Batch Processing...")
    batch_range = range(start_idx, total_count, BATCH_SIZE)
    
    for i in tqdm(batch_range, desc="Processing Batches"):
        end_idx = min(i + BATCH_SIZE, total_count)
        
        batch_comments = all_comments[i:end_idx]
        batch_df_slice = df_full.iloc[i:end_idx].copy()

        
        formatted_prompts = [prompt_template.format(comment_text=c) for c in batch_comments]

        outputs = text_generator(
            formatted_prompts,
            max_new_tokens=150, 
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1, 
            do_sample=True
        )

        batch_preds = []
        for idx, out in enumerate(outputs):
            result_obj = out[0] if isinstance(out, list) else out
            full_text = result_obj['generated_text']
            new_text = full_text.replace(formatted_prompts[idx], '').strip()
            
            pred = parse_model_output(new_text)
            batch_preds.append(pred)
          
        batch_df_slice['predicted_label'] = batch_preds
        write_header = not os.path.exists(results_path) or (start_idx == 0 and i == 0)
        
        batch_df_slice.to_csv(results_path, mode='a', header=write_header, index=False)

    print("\nInference job complete.")

    # Saving the results
    try:
        final_df = pd.read_csv(results_path, engine='python')
        
        if 'level-2' in final_df.columns:
            print("Generating metrics report...")
            
            y_true = final_df['level-2'].astype(int)
            y_pred = final_df['predicted_label'].astype(int)
            
            # Save text report
            report = classification_report(y_true, y_pred, zero_division=0)
            with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
                f.write(report)
            print(report)
          
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
            print("Saved confusion matrix image.")
            
        else:
            print("Info: 'level-2' column not found, skipping accuracy metrics.")
            
    except Exception as e:
        print(f"Warning: Could not generate final report: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Bias Detection: Zero-Shot Classification")
    parser.add_argument("--input_csv", type=str, required=True, help="Pathway to dataset") # Dataset file path
    parser.add_argument("--output_dir", type=str, default="./results", help="Results directory") # Output Directory
    parser.add_argument("--gpu_id", type=int, default=1, help="GPU ID")

    args = parser.parse_args()
    main(args)
