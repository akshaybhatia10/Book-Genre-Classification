from flask import Flask, jsonify
from sklearn.externals import joblib
from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

data = pd.read_csv('book32-listing.csv',encoding = "ISO-8859-1")

columns = ['Id', 'Image', 'Image_link', 'Title', 'Author', 'Class', 'Genre']
data.columns = columns

books = pd.DataFrame(data['Title'])
author = pd.DataFrame(data['Author'])
genre = pd.DataFrame(data['Genre'])

data['Author'] = data['Author'].fillna('No Book')
data['Title'] = data['Title'].fillna('No Book')

feat = ['Genre']
for x in feat:
    le = LabelEncoder()
    le.fit(list(genre[x].values))
    genre[x] = le.transform(list(genre[x]))
    
data['everything'] = pd.DataFrame(data['Title'] + ' ' + data['Author'])

# from nltk.corpus import stopwords
# stop = list(stopwords.words('english'))

# def change(t):
#     t = t.split()
#     return ' '.join([(i) for (i) in t if i not in stop])

# data['everything'].apply(change)    

print (data['everything'][0] + " " + le.inverse_transform([0])[0])

vectorizer = TfidfVectorizer(min_df=2, max_features=70000, strip_accents='unicode',lowercase =True,
                            analyzer='word', token_pattern=r'\w+', use_idf=True, 
                            smooth_idf=True, sublinear_tf=True, stop_words = 'english')
vectors = vectorizer.fit_transform(data['everything'])   
print (vectors.shape) 

@app.route('/')
def form():
    return render_template('form_submit.html')

#return render_template('form_action.html', book=book)


@app.route('/predict', methods=['POST'])
def predict():
	s = request.form['book']
	text = []
	text.append(s)
	text[0] = text[0].lower()
	arr = (vectorizer.transform(text))
	clf = joblib.load('best.pkl')
	prediction = (clf.predict(arr))
	prediction = le.inverse_transform(prediction)[0]
	return jsonify({'Genre is ': (prediction)})

if __name__ == '__main__':
	clf = joblib.load('best.pkl')
	app.run(host='0.0.0.0')

