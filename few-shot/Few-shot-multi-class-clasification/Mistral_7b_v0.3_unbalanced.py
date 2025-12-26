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
import gc

HF_API_KEY = "" # Hugging_face Token
TARGET_GPU = ""  # Target GPU node
BATCH_SIZE = 4  
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3" # Model ID

EXAMPLE_COUNTS = {
    1: 12,  
    2: 6,   
    3: 12   
}
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DATA_PATH = "" # Base Data file path
INPUT_CSV_PATH = "" # Dataset file path
OUTPUT_DIR = "" # # Output Directory

RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "classification_results.csv")
SUPPORT_CSV_PATH = os.path.join(OUTPUT_DIR, "support_set_used.csv") 
REPORT_PATH = os.path.join(OUTPUT_DIR, "classification_report.txt")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

SYSTEM_PROMPT = """
You are an expert in analysing the severity of regional biases in social media comments about Indian states and regions. You are provided with comments that have already been identified as containing regional bias. Your task is to determine the severity level of the bias present.

Task: Classify the severity of the regional bias in the comment as "SEVERE" (3), "MODERATE" (2), or "MILD" (1).

Definitions (Check in this order):
- LEVEL 3 (SEVERE): Comments that are overtly hostile, hateful, or derogatory. These include usage of regional slurs, dehumanising language, calls for exclusion (e.g., "Go back to your state"), or statements that promote hatred/violence against a specific region or group.
- LEVEL 2 (MODERATE): Comments that contain explicit negative generalisations, mockery, or clearly biased assumptions about a region's culture, language, or people. The tone is critical or mocking but does not incite violence or use extreme profanity/slurs.
- LEVEL 1 (MILD): Comments that contain subtle stereotypes, "benevolent" or positive biases (e.g., "People from State X are always smart"), or minor negative generalisations that are not aggressive. These comments rely on low-level regional tropes without expressing hostility.

Step-by-Step Analysis Process:
Step 1: Analyze the Stereotype or Generalization
Think: What specific regional claim is being made?
- Is it a positive generalisation?
- Is it a negative stereotype? 

Step 2: Assess Tone and Intent
Evaluate the emotional weight of the words:
- Is the tone aggressive, hateful, or threatening? (Check for Level 3 first)
- Is the tone mocking, sarcastic, or condescending? (Check for Level 2)
- Is the tone casual or "matter-of-fact"? (Check for Level 1)

Step 3: Check for Escalating Factors
Look for specific triggers:
- For Level 3: Does it contain slurs? Does it question citizenship/belonging? Is it dehumanising?
- For Level 2: Does it imply one group is superior to another?

Step 4: Final Classification
Based on the analysis above, assign the severity score:
- 3: If the bias is abusive, hateful, or extreme.
- 2: If the bias is explicit and negative, but not abusive.
- 1: If the bias is subtle, positive, or non-hostile.

Your response must include a brief line of reasoning followed by the final classification in the format "Classification: [1, 2, or 3]".
"""

# Mapping definitions to examples to ensure the model learns "Why"
REASONING_MAP = {
    1: "The comment uses sarcasm or a subtle stereotype without overt hostility, matching the criteria for Severity 1.",
    2: "The comment makes a direct negative generalisation or insult about the region/people, matching the criteria for Severity 2.",
    3: "The comment contains aggressive, dehumanising language or slurs, matching the criteria for Severity 3."
}

def load_model_and_tokenizer(model_id: str, device: str):
    print(f"Loading model: {model_id}.")
    torch_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map=device
    )
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def create_prompt_and_test_sets(df: pd.DataFrame, example_counts: dict) -> tuple[str, pd.DataFrame]:
    if os.path.exists(SUPPORT_CSV_PATH):
        print(f"Loading existing support set from {SUPPORT_CSV_PATH}.")
        df_few_shot = pd.read_csv(SUPPORT_CSV_PATH)
    else:
        print(f"\nSelecting Inverse Balanced examples for support set.")
        print(f"Target Counts: {example_counts}")
        
        df_few_shot = pd.DataFrame()
        for label, count in example_counts.items():
            df_class = df[df['level-2'] == label]
            short_candidates = df_class[df_class['comment'].str.len() < 500]
            
            if len(short_candidates) >= count:
                class_sample = short_candidates.sample(n=count, random_state=42)
            else:
                class_sample = df_class.sample(n=min(count, len(df_class)), random_state=42)
                
            df_few_shot = pd.concat([df_few_shot, class_sample])
        
        print(f"Selected {len(df_few_shot)} total examples for the support set.")
        df_few_shot.to_csv(SUPPORT_CSV_PATH, index=False)

    all_examples = []
    for _, row in df_few_shot.iterrows():
        all_examples.append((str(row['comment']).strip(), int(row['level-2'])))
    
    random.shuffle(all_examples)
    
    examples_str = "Here are reference examples of how to analyze and classify the comments:\n\n"
    for comment, label in all_examples:
        reasoning = REASONING_MAP.get(label, "Matches definition.")
        examples_str += f"### Example\nComment: \"{comment}\"\nReasoning: {reasoning}\nClassification: {label}\n\n"
      
    df_test_set = df.drop(df[df['comment'].isin(df_few_shot['comment'])].index)
    
    return examples_str, df_test_set

def parse_response(response: str) -> tuple[str, int]:
    cleaned_response = re.sub(r'<.*?>', '', response, flags=re.DOTALL).strip()
    reasoning = cleaned_response.split("Classification:")[0].strip()
    
    match = re.search(r"Classification:\s*(\d)", cleaned_response)
    if match:
        prediction = int(match.group(1))
    else:
        nums = re.findall(r'\b(1|2|3)\b', cleaned_response)
        if nums:
            prediction = int(nums[-1])
        else:
            prediction = 2 
            
    return reasoning, prediction

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(TARGET_GPU if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer = load_model_and_tokenizer(MODEL_ID, device=TARGET_GPU)
    
    print(f"\nLoading dataset from {INPUT_CSV_PATH}.")
    df_full = pd.read_csv(INPUT_CSV_PATH)
    
    df_full['level-2'] = pd.to_numeric(df_full['level-2'], errors='coerce')
    df_full.dropna(subset=['comment', 'level-2'], inplace=True)
    df_full = df_full[df_full['level-2'].isin([1, 2, 3])]
    df_full['level-2'] = df_full['level-2'].astype(int)

    few_shot_prompt, df_inference = create_prompt_and_test_sets(
        df=df_full,
        example_counts=EXAMPLE_COUNTS
    )

    processed_count = 0
    if os.path.exists(RESULTS_CSV_PATH):
        print(f"Found existing results at {RESULTS_CSV_PATH}. Checking for resume.")
        try:
            existing_df = pd.read_csv(RESULTS_CSV_PATH)
            processed_count = len(existing_df)
            print(f"Resuming after {processed_count} already processed comments.")
        except Exception as e:
            print(f"Could not read existing CSV ({e}). Starting from scratch.")
            processed_count = 0
    else:
        pd.DataFrame(columns=['comment', 'true_label', 'predicted_label', 'reasoning']).to_csv(RESULTS_CSV_PATH, index=False)

    df_remaining = df_inference.iloc[processed_count:]
    print(f"Remaining comments to process: {len(df_remaining)}")

    print("\nStarting inference...")
    
    for i in tqdm(range(0, len(df_remaining), BATCH_SIZE), desc="Classifying"):
        batch_df = df_remaining.iloc[i:i+BATCH_SIZE]
        
        batch_prompts = []
        for _, row in batch_df.iterrows():
            comment_text = str(row['comment'])
            user_content = (
                f"{few_shot_prompt}"
                f"### Task\n"
                f"Analyze the following comment based on the definitions and examples above.\n"
                f"Comment: \"{comment_text}\""
            )
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
            templated_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_prompts.append(templated_prompt)

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(device)
        
        generated_ids = None 
        batch_results = []

        try:
            generated_ids = model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                max_new_tokens=512, 
                do_sample=False,      
                temperature=0.1,      
                pad_token_id=tokenizer.eos_token_id
            )
            decoded_responses = tokenizer.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            for idx, raw_response in enumerate(decoded_responses):
                reasoning, prediction = parse_response(raw_response)
                original_row = batch_df.iloc[idx]
                batch_results.append({
                    'comment': original_row['comment'],
                    'true_label': original_row['level-2'],
                    'predicted_label': prediction,
                    'reasoning': reasoning
                })
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nOOM Error in batch {i}. Clearing cache and skipping...")
                torch.cuda.empty_cache()
                gc.collect()
                continue 
            else:
                print(f"Runtime Error in batch: {e}")
        except Exception as e:
            print(f"General Error in batch: {e}")
        
        if batch_results:
            temp_df = pd.DataFrame(batch_results)
            temp_df.to_csv(RESULTS_CSV_PATH, mode='a', header=False, index=False)

        del inputs
        if generated_ids is not None:
            del generated_ids
        gc.collect()
        torch.cuda.empty_cache()

    print("\nInference complete. Generating final report.")
    if os.path.exists(RESULTS_CSV_PATH):
        final_df = pd.read_csv(RESULTS_CSV_PATH)
        y_true = final_df['true_label']
        y_pred = final_df['predicted_label']
        
        print("\nClassification Report (Classes 1, 2, 3):")
        report = classification_report(y_true, y_pred, target_names=['SEVERITY 1', 'SEVERITY 2', 'SEVERITY 3'], labels=[1,2,3], zero_division=0)
        print(report)
        
        with open(REPORT_PATH, 'w') as f:
            f.write(report)
        
        cm = confusion_matrix(y_true, y_pred, labels=[1,2,3])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['SEV 1', 'SEV 2', 'SEV 3'], 
                    yticklabels=['SEV 1', 'SEV 2', 'SEV 3'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix (Inverse Balanced)')
        plt.savefig(CONFUSION_MATRIX_PATH)
        print(f"Confusion Matrix saved to {CONFUSION_MATRIX_PATH}")

if __name__ == "__main__":
    main()
