import pandas as pd
import time
import csv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm  # For progress bar
import os

# Configure Gemini
genai.configure(api_key="")  # Add your API key here
model = genai.GenerativeModel('gemini-2.0-flash')
generation_config = {
    "temperature": 0.3,
    "max_output_tokens": 400,
}

# Define your prompt template
def format_prompt(text):
    return f"""**[Legal Expert Instruction]**
You are an expert in Indian law. Convert this complex legal text into simple English for common people:
{text}

**[Response Requirements]**
- Begin with "Simplified explanation: "
- Use bold (**) for key terms
- Add bullet points for lists
- Include 1 real-world example"""

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), 
       stop=stop_after_attempt(3))
def generate_simple_explanation(text):
    """Generate explanation using Gemini"""
    try:
        response = model.generate_content(
            format_prompt(text),
            generation_config=generation_config
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error: {e}")
        raise  # Re-raise for tenacity

def process_row(row, is_parquet=False):
    """Process a single row"""
    if is_parquet:
        instruction = f"Explain {row['act_title']} Section {row['section']}"
        input_text = row['law']
    else:
        instruction = f"Explain BNS Section {row['BNS Section']}"
        input_text = row['BNS Description']
    
    try:
        output = generate_simple_explanation(input_text)
        if output:
            return instruction, input_text, output
    except Exception as e:
        print(f"Failed to process {instruction}: {str(e)}")
    return None

def process_data(input_path, output_csv, test_mode=False, is_parquet=False):
    """Process data with parallel execution"""
    if is_parquet:
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    if test_mode:
        df = df.head(12)

    # Check if file exists before writing header
    file_exists = os.path.exists(output_csv)

    # Write header if file does not exist (prevents overwriting issues)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['instruction', 'input', 'output'])  # Correct header

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_row, row, is_parquet) for _, row in df.iterrows()]

        # Write results as they complete
        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
                result = future.result()
                if result:
                    writer.writerow(result)  # Ensure it's written as a row

    print(f"Dataset created successfully: {output_csv}")
if __name__ == "__main__":
    output_filename = "law_dataset.csv"
    test_mode = False  # Set to True for testing
    
    # Process IPC CSV first
    process_data("bns.csv", output_filename, test_mode)
    
    # Process laws parquet
    process_data("laws.parquet", output_filename, test_mode, is_parquet=True)
    
    print(f"Dataset created successfully: {output_filename}")
