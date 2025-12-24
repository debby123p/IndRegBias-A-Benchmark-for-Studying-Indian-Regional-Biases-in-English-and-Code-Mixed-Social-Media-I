import os
import pandas as pd
import re
import time
import gc
from tqdm import tqdm
import google.generativeai as genai
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORTANT: Replace with API key 
GOOGLE_API_KEY = "" 

INPUT_CSV_PATH = "" # Dataset file path
OUTPUT_DIR = "" # Output Directory
COMMENT_COLUMN_NAME = "comment"
GROUND_TRUTH_COLUMN_NAME = "is RB?" 
BATCH_SIZE = 16 

MODEL_ID = "gemini-2.5-pro" 

MODEL_PROMPT = """
You are an expert in identifying regional biases in social media comments about Indian states and regions. Your task is to classify whether a comment contains regional biases or not.

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

Your response must include a brief line of reasoning followed by the final classification in the format "Classification: [0 or 1]".
"""

def setup_environment():
    # Creates the output directory.
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

def configure_model():
    # Configures the Gemini API and initializes the model.
    print("Configuring Gemini API.")
    if not GOOGLE_API_KEY:
        raise ValueError("Google API Key is not set. Please provide your key.")
    
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_ID)
        print(f"Successfully initialized model: {MODEL_ID}")
        return model
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        raise

def parse_single_response(response_text):
    # Robustly parses a single model response text to ensure a 0 or 1 output.
    prediction = -1
    reasoning = response_text.split("Classification:")[0].strip() or response_text

    match = re.search(r'Classification:\s*([01])', response_text)
    if match:
        prediction = int(match.group(1))
    else:
        if re.search(r'REGIONAL BIAS|1', response_text, re.IGNORECASE):
            prediction = 1
        elif re.search(r'NON-REGIONAL BIAS|0', response_text, re.IGNORECASE):
            prediction = 0
    
    if prediction == -1:
        print(f"Warning: Could not parse model output. Defaulting to 0. Response: '{response_text}'")
        prediction = 0
        
    return prediction, reasoning

def classify_batch(comments, model):
    # Generates classifications for a batch of comments.
    results = []
    
    for comment in comments:
        prompt = f"{MODEL_PROMPT}\n\nComment: \"{comment}\"\n\nBased on your analysis, provide your reasoning and final classification."
        
        prediction = 0 
        reasoning = "Error: Max retries exceeded."
        max_retries = 3
        backoff_factor = 2
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                response_text = response.text
                prediction, reasoning = parse_single_response(response_text)
                break 
            except Exception as e:
                print(f"An error occurred during API call (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    print(f"Retrying in {wait_time} seconds.")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Defaulting this comment to 0.")
                    reasoning = f"Error after max retries: {e}"

        results.append((prediction, reasoning))
        time.sleep(1) 
        
    return results

def generate_evaluation_outputs(df):
    # Generates and saves the classification report and confusion matrix.
    y_true = df[GROUND_TRUTH_COLUMN_NAME].astype(int)
    y_pred = df['predicted_label'].astype(int)

    report = classification_report(y_true, y_pred, target_names=["NON-REGIONAL BIAS (0)", "REGIONAL BIAS (1)"], zero_division=0)
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report for model: {MODEL_ID}\n\n")
        f.write(report)
    print(f"\nClassification report saved to {report_path}")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["NON-REGIONAL BIAS (0)", "REGIONAL BIAS (1)"], 
                yticklabels=["NON-REGIONAL BIAS (0)", "REGIONAL BIAS (1)"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {MODEL_ID}')
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

def main():
    # Main function to orchestrate the entire classification process.
    setup_environment()
    model = configure_model()
    
    print(f"\nReading input CSV from: {INPUT_CSV_PATH}")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        return

    if COMMENT_COLUMN_NAME not in df.columns:
        raise ValueError(f"Comment column '{COMMENT_COLUMN_NAME}' not found in the CSV.")

    df[COMMENT_COLUMN_NAME] = df[COMMENT_COLUMN_NAME].astype(str).fillna("")
    comments_to_process = df[COMMENT_COLUMN_NAME].tolist()
    
    all_results = []
    
    print(f"\nStarting classification for {len(comments_to_process)} comments with batch size {BATCH_SIZE} (logical batches).")
    
    for i in tqdm(range(0, len(comments_to_process), BATCH_SIZE), desc="Classifying batches"):
        batch_comments = comments_to_process[i:i + BATCH_SIZE]
        batch_results = classify_batch(batch_comments, model)
        all_results.extend(batch_results)
        gc.collect()
    
    df['predicted_label'] = [res[0] for res in all_results]
    df['model_reasoning'] = [res[1] for res in all_results]

    output_csv_path = os.path.join(OUTPUT_DIR, "classification_results.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"\nClassification complete. Results saved to {output_csv_path}")
    
    if GROUND_TRUTH_COLUMN_NAME in df.columns:
        print("\nStarting evaluation...")
        generate_evaluation_outputs(df)
    else:
        print(f"\nGround truth column '{GROUND_TRUTH_COLUMN_NAME}' not found. Skipping evaluation.")

if __name__ == "__main__":
    main()
