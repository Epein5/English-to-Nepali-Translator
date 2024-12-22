import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import pandas as pd
import os

# file_path = '../Dataset/english-nepali.xlsx'
file_path = './Dataset/english-nepali.xlsx'

# Verify if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Current working directory: {os.getcwd()}")


def load_data(file_path):
    df = pd.read_excel(file_path)
    english_sentences = df['english_sent'].fillna('').tolist()
    nepali_sentences = df['nepali_sent'].fillna('').tolist()
      
    return english_sentences, nepali_sentences
english_sentences, nepali_sentences = load_data(file_path)


# Preprocessing the text
english_preprocessed = [simple_preprocess(doc) for doc in english_sentences]
nepali_preprocessed = [simple_preprocess(doc) for doc in nepali_sentences]

# Training the Word2Vec models
english_model = Word2Vec(sentences=english_preprocessed, vector_size=100, window=5, min_count=1, workers=4)
nepali_model = Word2Vec(sentences=nepali_preprocessed, vector_size=100, window=5, min_count=1, workers=4)

# os.makedirs('Models', exist_ok=True)

# Saving the models
english_model.save("Tokenizer/Models/english_word2vec.model")
nepali_model.save("Tokenizer/Models/nepali_word2vec.model")

# Loading the models
# english_model = Word2Vec.load("english_word2vec.model")
# nepali_model = Word2Vec.load("nepali_word2vec.model")
