import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the dataset text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # replace everything that is NOT a letter to an empty space
    review = review.lower() # make lowercase
    review = review.split() # split reviews into separate words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # only words not included in the stopwords set are considered
    review = ' '.join(review)
    corpus.append(review)

