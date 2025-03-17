from openai import OpenAI
import openai

token = ""   # You can get auth token from API Token section on TIR Dashboard

openai.api_key = token
openai.base_url = ""
instruction = "Explain this fir to me in simple words along with all the bns section explained properly"
input_text = """"Date & Time of FIR: 15/03/2025, 11:41 AM
Location: Dadar Railway Station, Platform No. 12, Mumbai
Complainant: Sanyukta Sanjay Shinde (Age 21)
Incident Date & Time: 15/03/2025, between 08:39 AM - 09:50 AM
Incident Details:
The complainant was traveling from Diva to Santa Cruz via Dadar.
She left her backpack on the luggage rack in a ladies’ second-class coach.
Upon reaching Santa Cruz, she realized she had forgotten the bag and immediately contacted railway authorities.
Upon returning to Dadar station, she found the bag missing, suspected to be stolen.
Stolen Items:
Dell Laptop (Serial No: 4T8N9S2) – Worth ₹30,000
Backpack (Black with red stripe)
Legal Action:
Case registered under Section 305(c) of the Bharatiya Nyaya Sanhita (BNS), 2023
Investigation assigned to Datta Shivajirao Bhise (Police Havaldar, ID: 586)
Current Status: Investigation in progress."
"""  # You can provide additional context if needed
formatted_prompt = f"""<|begin_of_text|>
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = token
openai_api_base = openai.base_url 
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(model="Kartik12/Law-fine-tune-Meta-Llama-3.1-8B",
                                      prompt=formatted_prompt,max_tokens=500,temperature=0)
print("Completion result:", completion.choices[0].text)
