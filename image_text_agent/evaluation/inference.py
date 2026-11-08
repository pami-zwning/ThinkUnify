import torch
import os
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen3VLForConditionalGeneration
from eval_utils import run_evaluation

# --- Configuration ---
# Replace with your actual model path (local or HF Hub) and image/video folders
# --- 1. Load Model and Processor ---
MODEL_PATH = "/mnt/afs/kongxiangli/workspace/image_text_agent/ms-swift/output/v0-20251126-140045/checkpoint-33"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_PATH,trust_remote_code=True)
print("Model and processor loaded.")


# --- 2. Define Question and Image Path ---
question_text = '''
What is the material of the glove?
A. rubber
B. cotton
C. kevlar
D. leather
Answer with the option's letter from the given choices directly.'''
image_path = "./evaluation/0_0.png"

# --- 3. Generate Model Output ---
print("Generating model output...")
final_assistant_response, final_answer ,conversation_history = run_evaluation(question_text, image_path, model, processor)

print("\nFinal Answer:\n", final_answer)
print('conversation_history:',conversation_history)
    
