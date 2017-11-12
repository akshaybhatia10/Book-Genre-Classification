# Importing all libraries
import tensorflow as tf
import numpy as np
import re
from collections import Counter
import random
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import pickle
import pandas as pd

def read_data(book):
	'''
	params: text_file
	Load and extract the text file. 
	Return the text
	'''


	return text

def preprocess_text(book):
	'''
	clean dataset - replace numbers and stopwords with " " 
	'''	
	book = re.sub("[^a-zA-Z]", " ", book)
	words = book.lower().split()
	words = [w for w in words if w not in noise]
	all_text = ' '.join(words)
	return all_text 

def clean_text(path):
	'''
	Accepts books and returns cleaned titles
	'''
	data = pd.read_csv(path, encoding = "ISO-8859-1")
	columns = ['Id', 'Image', 'Image_link', 'Title', 'Author', 'Class', 'Genre']
	data.columns = columns
	books = []
	for i, book in enumerate(data['Title']):
		books.append(preprocess_text(book))

	return books

noise = set(stopwords.words("english"))

if __name__ == '__main__':
	books = clean_text('data/book32-listing.csv')
	print (len(books))
	print (books[:5])

