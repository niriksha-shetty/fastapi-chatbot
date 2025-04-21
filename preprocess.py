import re
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
nltk.download('punkt_tab')

def preprocess_text(text):
    # Remove special characters and extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9., ]', '', text)
    
    # Tokenize sentences
    sentences = sent_tokenize(text)
    
    return sentences

# Load extracted text
with open("Galethird.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

cleaned_sentences = preprocess_text(raw_text)

# Save structured data
with open("cleaned_Galethird.txt", "w", encoding="utf-8") as f:
    for sent in cleaned_sentences:
        f.write(sent + "\n")
