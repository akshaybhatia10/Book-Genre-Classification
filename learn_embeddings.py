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


def build_vocab(processed_books, duplicates=True):
	'''
	Return a list of all words: keep duplicates
	'''
	words = []
	for book in books:
		word = book.split()
		for w in word:
			words.append(w)
	vocab = list(set(words)) if not duplicates else words

	return vocab		

def prepare_data(words):
	'''
	Create lookup tables to map words to indices and viceversa
	'''
	count = Counter(words)
	sorted_count = sorted(count, key=count.get, reverse=True)
	vocab_to_int = {word:i for i, word in enumerate(sorted_count)} 
	int_to_vocab = {i:word for word, i in vocab_to_int.items()}

	return vocab_to_int, int_to_vocab

def sampling(data_to_ints, threshold=1e-5):
	'''
	Removing words such as 'the', 'of' etc.
	'''
	total = len(data_to_ints)
	index_count = Counter(data_to_ints)
	word_frequency = {word: count/total for word, count in index_count.items()}
	drop_prob = {word: 1-np.sqrt(threshold/word_frequency[word]) for word in index_count}
	final_words = [word for word in data_to_ints if random.random() < (1 - drop_prob[word])]

	return final_words

def window_words(words, index, window_size=5):
	'''
	returns a list of words in the window around the index
	'''
	s = np.random.randint(1, window_size+1)
	first = index - s if (index - s) > 0 else 0
	last = index + s
	close_words = set(words[first:index] + words[index+1:last+1])

	return list(close_words)

def generate_batches(words, batch_size, window_size=5):
	'''
	Returns batches(inputs and targets)
	'''
	batches = len(words) // batch_size
	words = words[:batch_size*batch_size]
	for i in range(0, len(words), batch_size):
		inputs, targets = [], []
		batch = words[i:i+batch_size]
		for ii in range(len(batch)):
			x = batch[ii]
			y = window_words(batch, ii, window_size)
			targets.extend(y)
			inputs.extend([x]*len(y))
		yield inputs, targets	

def plot_embeddings(embeddings, targets, file='vis_book_embeddings.png'):
	'''
	Plot the n-dimensional embeddings
	'''
	plt.figure(figsize=(18,18))
	for (i, target) in enumerate(targets):
		x, y = embeddings[i, :]
		plt.scatter(x, y)
		plt.annotate(target, xy=(x,y), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
	plt.savefig(file)


noise = set(stopwords.words("english"))

# Model Hyperparameters
batch_size = 512
embedding_size = 128
window_size = 5
sampled = 100
valid_size = 16
valid_window = 100
epochs = 10


if __name__ == '__main__':
	books = clean_text('data/book32-listing.csv')
	#print (len(books))
	#print (books[:5])
	
	vocab = build_vocab(books)
	vocab_to_int, int_to_vocab = prepare_data(vocab)
	#print (vocab_to_int)
	
	vocab_size = len(int_to_vocab)
	data_to_ints = [vocab_to_int[word] for words in books for word in words.split()]
	
	#print (data_to_ints[:10])
	#for i in data_to_ints[:10]:
	#	print (int_to_vocab[i])
	
	final_words = sampling(data_to_ints)

		## Build the network
	inputs = tf.placeholder(tf.int32, shape=(None))
	targets = tf.placeholder(tf.int32, shape=(None, None))


	embeddings = tf.Variable(tf.random_uniform((vocab_size, embedding_size), -1, 1))
	embed = tf.nn.embedding_lookup(embeddings, inputs)

	weights = tf.Variable(tf.truncated_normal((vocab_size, embedding_size)))
	biases = tf.Variable(tf.zeros(vocab_size))

	softmax_loss = tf.nn.sampled_softmax_loss(weights, biases, targets, embed, sampled, vocab_size)
	loss = tf.reduce_mean(softmax_loss)

	optimizer = tf.train.AdamOptimizer().minimize(loss)

	valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
	valid_examples = np.append(valid_examples, random.sample(range(1000, 1000+valid_window), valid_size//2))
	valid_set = tf.constant(valid_examples, dtype=tf.int32)

	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_set)
	similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

	# Training the model
	init = tf.global_variables_initializer()
	session = tf.Session()
	saver = tf.train.Saver()
	
	with session as sess:
		sess.run(init)
		i = 1
		total_loss = 0
		for epoch in range(epochs+1):
			start_epoch = time.time()
			batches = generate_batches(final_words, batch_size, window_size)
			start = time.time()
			for x, y in batches:
				l, _ = sess.run([loss, optimizer], feed_dict={inputs:x, targets:np.array(y)[:, None]})
				total_loss += l

				if i % 200 == 0:
					end = time.time()
					print ("Epoch: {}/{}, Iteration: {}, Average loss: {:.4f}, Time/batch: {}".format(epoch, epochs, i, total_loss, (end-start)))

					total_loss = 0
					start = time.time()

				if i % 500 == 0:
					sim = similarity.eval()
					for j in range(valid_size):
						valid_word = int_to_vocab[valid_examples[j]]
						top = 8
						nearest = (-sim[j, :]).argsort()[1:top+1]		
						string = 'Nearest to {} : '.format(valid_word)	
						for k in range(top):
							closest = int_to_vocab[nearest[k]]
							string = string + " " + closest
						print (string)
				i += 1

			#end_epoch = time.time()
			#print ('Total Epoch Time : {:.4}'.format(end_epoch-start_epoch))

		saved_At = saver.save(sess, "embeddings/final_100.ckpt")
		final_embeddings = sess.run(normalized_embeddings)					

		with open('embeddings/embeddings.txt', 'wb') as f:
			print ("Saving Embeddings...")
			pickle.dump([final_embeddings, int_to_vocab], f, -1)

				# Plotting learned vector representations using TSNE
		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
		new_embeddings = tsne.fit_transform(final_embeddings[:500, :])
		choosen_targets = [int_to_vocab[i] for i in range(500)]
		plot_embeddings(new_embeddings, choosen_targets)	

