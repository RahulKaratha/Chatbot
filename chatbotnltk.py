import nltk
from nltk.chat.util import Chat,reflections

pairs = [
    (r"my name is (.*)", ["Hello %1, how are you today?"]),
    (r"hi|hello|hey", ["Hello!", "Hi there!", "Hey!"]),
    (r"what is your name?", ["I am a chatbot."]),
    (r"how are you?", ["I'm doing well, thank you!"]),
    (r"sorry (.*)", ["It's okay!", "No problem."]),
]

chatbot = Chat(pairs, reflections)

print("Hi, I'm your chatbot. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = chatbot.respond(user_input)
    print("Bot:", response)
