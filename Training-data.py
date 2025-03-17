import wandb
from datasets import Dataset
from transformers import AutoTokenizer

def load_and_prepare_data():
    # Initialize W&B
    wandb.init(project="Law-dataset-finetunning", job_type="data-preparation")
    
    # Load dataset artifact
    artifact = wandb.use_artifact("Law-finetunning-dataset:latest")
    artifact_dir = artifact.download()
    
    # Load dataset
    with open(f"{artifact_dir}/law_dataset.json") as f:
        alpaca_data = json.load(f)
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(alpaca_data)
    
    # Formatting function from the book
    def format_samples(samples):
        EOS_TOKEN = tokenizer.eos_token
        formatted_texts = []
        
        for instruction, input_text, output in zip(
            samples['instruction'],
            samples['input'],
            samples['output']
        ):
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}{EOS_TOKEN}"""
            
            formatted_texts.append(prompt)
        
        return {'text': formatted_texts}
    
    # Apply formatting
    tokenizer = tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    token="",
    trust_remote_code=True
)
    dataset = dataset.map(
        format_samples,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split dataset (95% train, 5% test)
    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    
    # Log dataset statistics
    wandb.log({
        "train_samples": len(split_dataset['train']),
        "test_samples": len(split_dataset['test'])
    })
    
    return split_dataset

# Usage example
if __name__ == "__main__":
    dataset = load_and_prepare_data()
    
    # To access splits
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # Continue with training...
    wandb.finish()
