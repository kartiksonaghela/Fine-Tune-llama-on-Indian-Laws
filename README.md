# Fine-Tune-llama-on-Indian-Laws

# Legal AI Agent

## Overview

Legal AI Agent is an AI-powered tool designed to simplify complex legal texts. It leverages large language models to break down legal jargon, interpret documents, and provide clear, accessible legal explanations.

## Features

- **Legal Text Interpretation**: Converts complex legal jargon into simplified explanations.
- **Query-Based Legal Assistance**: Answers legal queries based on Indian laws.
- **Document Analysis**: Extracts and explains key details from legal documents such as FIRs.
- **Fine-Tuned LLM**: Trained on legal datasets using LoRA fine-tuning for efficient processing.

## Dataset

The model is trained on:

- **Bharatiya Nyaya Sanhita (BNS) Sections**: 563 sections of Indian penal provisions.
- **Public & Administrative Laws**: 34,000 Indian acts covering various legal frameworks.

## Model Training

The model was fine-tuned using Unsloth’s FastLanguageModel with LoRA:

- **Base Model**: Meta-Llama-3.1-8B
- **Fine-Tuning**: LoRA applied to key transformer modules.
- **Training Hyperparameters**:
  - Learning Rate: 3e-4
  - Batch Size: 4
  - Epochs: 3
  - Optimizer: AdamW 8-bit
  - Gradient Accumulation: 4

## Infrastructure

The training and inference are hosted on **E2E Networks Cloud**:

- **Training Setup**:
  - GPU: A100 80GB
  - CPU: 16 Cores
  - RAM: 115GB
  - Pricing: ₹226/hr (Hourly Billing)
- **Inference Deployment**:
  - Framework: vLLM
  - Model: Hugging Face ID `Kartik12/Law-fine-tune-Meta-Llama-3.1-8B`
  - Server: 1 GPU (24GB Memory), 25 CPUs, 110GB RAM
  - Pricing: ₹50/hr (Hourly Billing)

## API Usage

The model is deployed as a REST API. Example request:

```json
{
  "instruction": "Explain BNS Section 123 in simple words.",
  "input": "Original legal text of BNS Section 123..."
}
```

Expected Response:

```json
{
  "output": "Simplified explanation: [simplified text...]"
}
```



## Future Enhancements

- Expanding dataset coverage to more legal frameworks.
- Adding multilingual support.
- Implementing real-time legal case analysis.

## Credits

- **Developed by**: Kartik
- **Hosted on**: [E2E Networks](https://www.e2enetworks.com/)

