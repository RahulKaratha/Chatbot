from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"  # You can use small, medium, or large models
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a response based on user input
def generate_response(user_input, chat_history_ids=None):
    # Encode the user input and add it to the chat history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input to the conversation history
    if chat_history_ids is not None:
        chat_history_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        chat_history_ids = new_input_ids

    # Generate a response
    response_ids = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the response and return it
    response = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# Chatbot interaction loop
chat_history = None
print("Start chatting with the bot! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response, chat_history = generate_response(user_input, chat_history)
    print(f"Bot: {response}")
