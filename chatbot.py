import json
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load intents from a JSON file
def load_intents(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Preprocess user input
def preprocess_input(user_input):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Tokenize, remove stopwords, and lemmatize
    tokens = word_tokenize(user_input.lower())
    processed = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return processed

# Get response based on intents
def get_response(intents, user_input):
    user_words = preprocess_input(user_input)
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            pattern_words = preprocess_input(pattern)
            if all(word in user_words for word in pattern_words):
                return random.choice(intent["responses"])
    return "I'm sorry, I don't understand. Can you rephrase?"

# Main chatbot function
def chatbot():
    print("Chatbot: Hello! How can I help you? (type 'quit' to exit)")

    intents = load_intents("intents.json")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        response = get_response(intents, user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
