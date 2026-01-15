import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import nltk

# List of all resources your preprocessor needs
resources = {
    'corpora/stopwords': 'stopwords',
    'tokenizers/punkt': 'punkt',
    'corpora/wordnet': 'wordnet',
    'corpora/omw-1.4': 'omw-1.4'  # Required for modern WordNet
}

for path, package in resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package)

class FinancialTextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def extract_tickers(self, text):
        return re.findall(r'\$[A-Z]+', str(text))

    def clean_text(self, text):
        text = str(text)
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        if not text.strip(): return ''
        
        # Tokenize & Lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens 
                  if t not in self.stop_words and len(t) > 2]
        return ' '.join(tokens)

# The exact 20 topics from your notebook
LABEL_MAPPING = {
    0: "Analyst Update", 1: "Fed | Central Banks", 2: "Company | Product News",
    3: "Treasuries | Corporate Debt", 4: "Dividend", 5: "Earnings",
    6: "Energy | Oil", 7: "Financials", 8: "Currencies",
    9: "General News | Opinion", 10: "Gold | Metals | Materials", 11: "IPO",
    12: "Legal | Regulation", 13: "M&A | Investments", 14: "Macro",
    15: "Markets", 16: "Politics", 17: "Personnel Change",
    18: "Stock Commentary", 19: "Stock Movement"
}