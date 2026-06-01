# pip install flask transformers torch sentencepiece

from flask import Flask, request, jsonify, render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re

app = Flask(__name__)

# Load the saved model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./chatbot_model")
tokenizer = T5Tokenizer.from_pretrained("./chatbot_model")

device = model.device




# Clean the text by removing unwanted characters
def clean_text(text):
    text = re.sub(r'\r\n', ' ', text)  # Remove carriage returns and line breaks
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'<.*?>', '', text)  # Remove any XML tags
    text = text.strip().lower()  # Strip and convert to lower case
    return text

# Chatbot function
def chatbot(dialogue):
    dialogue = clean_text(dialogue)  # Assuming clean_text is defined
    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, padding="max_length", max_length=250)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        inputs["input_ids"],
        max_length=250,
        num_beams=4,
        early_stopping=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Rendering Index root
@app.route("/")
def index():
    return render_template("index.html")

# API endpoint
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    response = chatbot(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
