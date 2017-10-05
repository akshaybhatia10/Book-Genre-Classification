# Book Genre Classification
 

### About 

This project was a part of Spikeway Technologies AI Lab(SAIL) and discusses the classification of books based only on title, without prior knowledge or context of author and origin. The total number of genres for the books to be classified into is 32. The notebooks focus on achieving this task using different Machine Learning and Deep Learning techniques. 

**_Basic_Bag_of_Words_model.ipynb_** does a very simple exploration and uses a basic feed forward network.

**_Best_TFIDF-Vectorizer_model.ipynb_** uses Logistic Regression, Multinominal NaiveBayes, Multi Layer Perceptron and XGBoost.

**_RNN_LSTM_using_Glove_vectors.ipynb_** uses Deep Learning concepts such as Recurrent Neural Networks(RNN) and Long Short Term Memory(LSTM).

A live demo hosted using [Heroku](https://heroku.com) is available [here](https://book-genre-classification.herokuapp.com). This webapp uses the pre-saved Logistic Regression model for inference.  

### Install 

This project requires **Python 3** and the following Python libraries installed:

- [sklearn](http://scikit-learn.com/)
- [Tensorflow](http://tensorflow.org/)
- [pandas](pandas.pydata.org/)
- [Numpy](http://numpy.org/)
- [Scipy](http://scipy.org/)
- [Matplotlib](https://matplotlib.org/) 
- [tflearn](https://tflearn.org/)



You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

You could just install [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. 


### Run

In a terminal or command window, navigate to the top-level project directory  (that contains this README) and run one of the following commands (For example to run Basic.ipynb):


```bash
ipython notebook Best_TFIDF-Vectorizer_model.ipynb
```  
or
```bash
jupyter notebook Best_TFIDF-Vectorizer_model.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.


To deploy the live working [demo](https://book-genre-classification.herokuapp.com), first clone the repo. Make sure you have a Heroku account.Then follow the following commands:

```bash
cd appweb
heroku login
git init
git add .
git commit -m "done"
heroku create    		### create heroku remote
git remote -v 
heroku git:remote -a <your-app-name(as created by heroku)>
git push heroku master
``` 


### Dataset

This dataset contains book cover images, title, author, and subcategories for each respective book.

Format:
```
"[AMAZON INDEX (ASIN)}","[FILENAME]","[IMAGE URL]","[TITLE]","[AUTHOR]","[CATEGORY ID]","[CATEGORY]"
```

Example:
```
"1588345297","1588345297.jpg","http://ecx.images-amazon.com/images/I/51l6XIoa3rL.jpg","With Schwarzkopf: Life Lessons of The Bear","Gus Lee","1","Biographies & Memoirs"
"1404803335","1404803335.jpg","http://ecx.images-amazon.com/images/I/51UJnL3Tx6L.jpg","Magnets: Pulling Together, Pushing Apart (Amazing Science)","Natalie M. Rosinsky","4","Children's Books"
```

32 Genres, 207,572 Books

|Label|Category Name|Size|
|---|---|---|
|0|Arts & Photography|6,460|
|1|Biographies & Memoirs|4,261|
|2|Business & Money|9,965|
|3|Calendars|2,636|
|4|Children's Books|13,605|
|5|Comics & Graphic Novels|3,026|
|6|Computers & Technology|7,979|
|7|Cookbooks, Food & Wine|8,802|
|8|Crafts, Hobbies & Home|9,934|
|9|Christian Books & Bibles|9,139|
|10|Engineering & Transportation|2,672|
|11|Health, Fitness & Dieting|11,886|
|12|History|6,807|
|13|Humor & Entertainment|6,896|
|14|Law|7,314|
|15|Literature & Fiction|7,580|
|16|Medical Books|12,089|
|17|Mystery, Thriller & Suspense|1,998|
|18|Parenting & Relationships|2,523|
|19|Politics & Social Sciences|3,402|
|20|Reference|3,268|
|21|Religion & Spirituality|7,559|
|22|Romance|4,291|
|23|Science & Math|9,276|
|24|Science Fiction & Fantasy|3,800|
|25|Self-Help|2,703|
|26|Sports & Outdoors|5,968|
|27|Teen & Young Adult|7,489|
|28|Test Preparation|2,906|
|29|Travel|18,338|
|30|Gay & Lesbian|1,339|
|31|Education & Teaching|1,664|
