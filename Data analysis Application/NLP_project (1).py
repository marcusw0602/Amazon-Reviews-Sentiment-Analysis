from pandas.io.parsers import PythonParser
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import neattext.functions as nfx
from textblob import TextBlob
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import wordcloud
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
warnings.filterwarnings('ignore')
menu = ['Home','Exploritory Data Analysis','Machine Learning Model']

choice = st.sidebar.selectbox("Please make your selection below",menu)
df = pd.read_csv(r"C:\Users\marcu\OneDrive\Desktop\Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
def main():
    if choice == 'Home':
        st.title("Amazon Sentitment Analysis Data App")
        st.text('This App was built by Demarcus, Himanshu, and Diego')

        if st.checkbox("Check to show our groups Project Proposel"):
            st.markdown("""
**The topic of study Analysis based on Consumer Reviews of Amazon Products**

**Background** –

Product reviews are becoming more important with the evolution of traditional brick and mortar retail stores to online shopping. Consumers are posting reviews directly on product pages in real time. With the vast amount of consumer reviews, this creates an opportunity to see how the market reacts to a specific product. We will be attempting to see if we can analyze and then predict the sentiment based on a product review using machine learning tools.
 
**1.   Consumer Reviews of Amazon Products** -
 
This is a list of over 34,000 consumer reviews for Amazon products like the Kindle, Fire TV Stick, and more provided by Datafiniti's Product Database. The dataset includes basic product information, rating, review text, and more for each product.
Note that this is a sample of a large dataset. The full dataset is available through Datafiniti.
This dataset also contains image and urls for the product for each review, this information may be useful for visualization purposes so that the product itself can be seen when making a UI for the project.
 
**Dataset for Analysis**

Source- Direct download https://www.kaggle.com/datafiniti/consumer-reviews-of-amazon-products/download

Website-https://www.kaggle.com/datafiniti/consumer-reviews-of-amazon-products?select=Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv
https://developer.datafiniti.co/docs/product-data-schema

**Hypothesis Testing and Identification of Gaps in the Present Recommending System Literature**
 
- Which product categories have lower reviews or which products have higher reviews ?
- Which products should be kept or which one can be  dropped from Amazon’s product roster (which ones are junk?)
- Can we associate positive and negative sentiments with each product in Amazon’s Catalog
- By using Sentiment analysis, can we predict scores for reviews based on certain words
- Can we say that if Product X is highly rated on the market, it seems most people like its lightweight sleek design and fast speeds. Most products that were associated with negative reviews seemed to indicate that they were too heavy and they couldn’t fit them in the bags. So  using the eda and above mentioned questions can we suggest that next generation models for e-readers are lightweight and portable, based on this data we’ve looked at.

**Stratifying the data**

- Since the majority of reviews are positive (5 stars), we will need to do a stratified split on the reviews score to ensure that we don’t train the classifier on imbalanced data
- To use sklearn’s Stratified ShuffleSplit class, we’re going to remove all samples that have NAN in review score, then convert all review scores to integer datatype.

**Data Web Application built using Streamlib**

We want to present our project in a way that can show our findings in an easily understandable and presentable way. We all agreed that something presenting a Jupyter notebook is difficult to present due to the formatting of notebooks. Instead we opted to develop a web application to address these needs.  

This application is built on the python library streamlit. This is a popular upcoming package used for creating seamless and simplistic data applications. The application will feature a home page describing our project in its entity, a EDA section with some data visualizations, and lastly a section for the machine learning section where we will try and predict whether or not a review is positive or negative as well as the sentiment level. 

We will also include a section to where you can upload any data file into the application to where it can spit out the EDA, Visualizations, and Machine learning section out. 

One of the reasons why we opted to use a dataset with image urls is so that we have the ability to display the product that is associated with a review. An example of this could be when making a sentiment prediction of a specified product so that a user can see the product itself rather than just a string of words.

**Steps-**
- data exploration
- EDA        
- Sentiment Analysis
- Extract Features
- Building a Pipeline from the Extracted Feature
- Test Different Model
- Fine tuning
- Detailed Performance Analysis
- Developing web application for UI

**What outcomes you are expecting**

- We’re expecting that the information we find in the text reviews of each product will be rich enough to train a sentiment analysis classifier with accuracy (hopefully) > 70%
- To be able to extract the various words that are associated with positive and negative sentiment in these reviews.
- Using the classifier to determine whether product reviews outside of the training dataset have a positive or negative sentiment.
- Observe the difference of sentiment between similar products and what is the reason for the difference.
- Observe which categories of products have the most positive or negative sentiment.
- Finding common complaints or praises that determine what consumers are looking for so that future products can use that information to improve their products.
- Developing a UI that will display our results in an easily presentable manner.

""")
        st.subheader("Please use the slide bar on the left hand side of the screen to navigate the application.")

    if choice == 'Exploritory Data Analysis':
        st.title('Exploritory Data Analysis')
        df = pd.read_csv(r"C:\Users\marcu\OneDrive\Desktop\Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
        df['reviews.text'] = df['reviews.text'].apply(nfx.remove_stopwords)
        
        st.markdown("Hello everyone welcome to this section's EDA for our Amazon review analysis. Please see the below message on the background of the data we are looking at.")
        
        st.cache()
        number = st.slider('Please slide the number to confirm the number of rows you would like to display from the top:',max_value=len(df))
        st.write(df.head(number))
        
        st.subheader('This is the current list of columns from the dataset' +' ('+str(len(df.columns))+')')
        st.code((['id', 'dateAdded', 'dateUpdated', 'name', 'asins', 'brand',
       'categories', 'primaryCategories', 'imageURLs', 'keys', 'manufacturer',
       'manufacturerNumber', 'reviews.date', 'reviews.dateSeen',
       'reviews.didPurchase', 'reviews.doRecommend', 'reviews.id',
       'reviews.numHelpful', 'reviews.rating', 'reviews.sourceURLs',
       'reviews.text', 'reviews.title', 'reviews.username', 'sourceURLs']))
        
        st.subheader('Based on the descriptive statistics below, we see the following:')
        st.markdown('''Average review score of 4.51, with low standard deviation Most review are positive from 2nd quartile onwards The average for number of reviews helpful (reviews.numHelpful) is 0.5 but high standard deviation The data are pretty spread out around the mean, and since can’t have negative people finding something helpful, then this is only on the right tail side The range of most reviews will be between 0-13 people finding helpful (reviews.numHelpful) The most helpful review was helpful to 621 people This could be a detailed, rich review that will be worth looking at.''')
        st.write(df.describe())

        if st.checkbox('Please check the box to view some other descriptive analysis'):
            asins_unique = len(df["asins"].unique())
            st.write("Number of Unique ASINs: " + str(asins_unique))
            st.write(df["asins"].unique())
            st.write('Based on the information below: Drop reviews.userCity, reviews.userProvince, reviews.id, and reviews.didPurchase since these values are floats (for exploratory analysis only) Not every category have maximum number of values in comparison to total number of values reviews.text category has non missing data (28332/28332) -> its good.')
            st.write(df.dtypes)
            st.write('Please see the following top 8 products below.')
            st.write(df['name'].value_counts().nlargest(8))
            st.write("Here you can see the most frequent words showing within the dataset below:")
            freq = pd.Series(' '.join(df['reviews.text']).split()).value_counts()[:20]
            st.write(freq)

        def get_sentiment(text):
            blob = TextBlob(text)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_subjectivity = blob.sentiment.subjectivity
            if sentiment_polarity > 0:
                sentiment_label = 'Positive'
            elif sentiment_polarity < 0:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'
            result = {'polarity':sentiment_polarity,'subjectivity':sentiment_subjectivity,'sentiment':sentiment_label}
            return result
        
        df['Sentiment_results'] = df['reviews.text'].apply(get_sentiment)
        df = df.join(pd.json_normalize(df['Sentiment_results']))
        
        st.subheader('Sentiment Category for Amazon product reviews')
        st.write(plt.figure(figsize=(20,10)),sns.countplot(df['sentiment']))
        st.markdown('As you can see there above there is a large number of positive product reviews in this dataset')
        st.write(df['sentiment'].value_counts())
        st.write(df["sentiment"].value_counts()/len(df))


        positive_words = df[df['sentiment'] == 'Positive']['reviews.text']
        negative_words = df[df['sentiment'] == 'Negative']['reviews.text']
        neutral_words = df[df['sentiment'] == 'Neutral']['reviews.text']

        positive_list = positive_words.apply(nfx.remove_stopwords).tolist()
        negative_list = negative_words.apply(nfx.remove_stopwords).tolist()
        neutral_list = neutral_words.apply(nfx.remove_stopwords).tolist()

        pos_tokens = [token for line in positive_list  for token in line.split()]
        neg_tokens = [token for line in negative_list  for token in line.split()]
        neut_tokens = [token for line in neutral_list  for token in line.split()]

        def get_tokens(docx,num=30):
            word_tokens = Counter(docx)
            most_common = word_tokens.most_common(num)
            result = dict(most_common)
            return result
        
        most_common_pos_words = get_tokens(pos_tokens)
        most_common_neg_words = get_tokens(neg_tokens)
        most_common_neut_words = get_tokens(neut_tokens)

        neg_df = pd.DataFrame(most_common_neg_words.items(),columns=['words','scores'])
        pos_df = pd.DataFrame(most_common_pos_words.items(),columns=['words','scores'])
        neu_df = pd.DataFrame(most_common_neut_words.items(),columns=['words','scores'])
        
        st.markdown('We broke down the dataset further into three set containing only reviews that match its positive, negative, neutral tones.')
        
        st.cache()
        if st.checkbox('Positive'):
            number = st.slider('Please slide the number to confirm the number of rows you would like to display from the top:',max_value=len(positive_words))
            st.write(positive_words.head(number))
            st.markdown('What we also found interesting is that the words that were used in the positive reviews were all words that reflectivity')
            st.write(get_tokens(pos_tokens))
            st.markdown('The Frequency of Positive words')
            st.write(plt.figure(figsize=(20,10)),sns.barplot(x='words',y='scores',data=pos_df),plt.xticks(rotation=45),plt.show())

        if st.checkbox('Negative'):
            number = st.slider('Please slide the number to confirm the number of rows you would like to display from the top:',max_value=len(negative_words))
            st.write(negative_words.head(number))
            st.markdown('What we also found interesting is that the words that were used in the negative reviews were all words that reflectivity')
            st.write(get_tokens(neg_tokens))
            st.markdown('The Frequency of Negative words')
            st.write(plt.figure(figsize=(20,10)),sns.barplot(x='words',y='scores',data=neg_df),plt.xticks(rotation=45),plt.show())

        if st.checkbox('Neutral'):
            number = st.slider('Please slide the number to confirm the number of rows you would like to display from the top:',max_value=len(neutral_words))
            st.write(neutral_words.head(number))
            st.markdown('What we also found interesting is that the words that were used in the neutral reviews were all words that reflectivity')
            st.write(get_tokens(neut_tokens))
            st.subheader('The Frequency of Neutral words')
            st.write(plt.figure(figsize=(20,10)),sns.barplot(x='words',y='scores',data=neu_df),plt.xticks(rotation=45),plt.show())
        
        st.cache()
        st.markdown("We have also broken down each review and assigned it, it's own sentiment analysis score see below")
        number = st.slider("Please slide to the desired row number to see the following review and it's sentiment score (note stopwords have been taken out):",max_value=len(df))
        df['reviews.text'] = df['reviews.text'].apply(nfx.remove_stopwords)
        review = df['reviews.text'].iloc[number]
        st.write(review)
        st.write(get_sentiment(review))

    if choice == 'Machine Learning Model':
        df = pd.read_csv(r"C:\Users\marcu\OneDrive\Desktop\Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
        st.title('Welcome to the Machine Learning Model')
        st.markdown('Before we explore the dataset we’re going to split it into training set and test sets Our goal is to eventually train a sentiment analysis classifier Since the majority of reviews are positive (5 stars), we will need to do a stratified split on the reviews score to ensure that we don’t train the classifier on imbalanced data To use sklearn’s Stratified ShuffleSplit class, we’re going to remove all samples that have NAN in review score, then covert all review scores to integer datatype')
        
        from sklearn.model_selection import StratifiedShuffleSplit
        Code='''        from sklearn.model_selection import StratifiedShuffleSplit 
        dataAfter = df.dropna(subset=["reviews.rating"]) 
        dataAfter["reviews.rating"] = dataAfter["reviews.rating"].astype(int)'''
        st.code(Code,language='python')
        st.markdown("Before {}".format(len(df)))
        dataAfter = df.dropna(subset=["reviews.rating"])
        st.markdown("After {}".format(len(dataAfter)))
        dataAfter["reviews.rating"] = dataAfter["reviews.rating"].astype(int)

        st.markdown('Stratified ShuffleSplit class')
        split = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
        for train_index, test_index in split.split(dataAfter, dataAfter["reviews.rating"]):
            strat_train = dataAfter.reindex(train_index)
            strat_test = dataAfter.reindex(test_index)
        Code1 = '''split = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
for train_index, test_index in split.split(dataAfter,dataAfter["reviews.rating"]): 
    strat_train = dataAfter.reindex(train_index)
    strat_test = dataAfter.reindex(test_index)'''
        st.code(Code1,language='python')

        if st.checkbox('Check to see if train/test sets were stratified proportionately in comparison to raw data'):
            st.write('Here is the size of the train dataset below:')
            st.write(len(strat_train))
            st.write('Here is the percentage of the review rating from the train set:')
            st.write(strat_train["reviews.rating"].value_counts()/len(strat_train))
            st.write('Here is the percentage of the review rating from the test set')
            st.write(strat_test["reviews.rating"].value_counts()/len(strat_test))
            st.write('As we can see above the review rating of five is sitting at 70% of the data set this will be interesting to see when we create our model')
        
        st.subheader('Hypothesis Question')
        st.markdown('Approach 1( We will also try to figure out if calculating sentiment for the text on the basis of rating(already present in the dataset) and reviews is better)')
        st.markdown('and')
        st.markdown('Approach 2(calculating sentiment for the text based on the polarity function from the textblob is better?')

        Approach = st.radio('To see  our first approach please select the first option, if you would like to see the second approach please select the second option',('Approach 1','Approach 2'))
        if Approach == 'Approach 1':
            st.write('To begin we are going to set the target variables in the dataset to reflect a more efficient view.')
            def sentiments(rating):
                if (rating == 5) or (rating == 4):
                    return "Positive"
                elif rating == 3:
                    return "Neutral"
                elif (rating == 2) or (rating == 1):
                    return "Negative"
            strat_train["Sentiment"] = strat_train["reviews.rating"].apply(sentiments)
            strat_test["Sentiment"] = strat_test["reviews.rating"].apply(sentiments)
            Code2 = '''def sentiments(rating):
    if (rating == 5) or (rating == 4):
        return "Positive"
    elif rating == 3:
        return "Neutral"
    elif (rating == 2) or (rating == 1):
        return "Negative"
    strat_train["Sentiment"] = strat_train["reviews.rating"].apply(sentiments)
    strat_test["Sentiment"] = strat_test["reviews.rating"].apply(sentiments)'''
            st.code(Code2,language='python')
            strat_train["Sentiment"][:20]
            st.write('See here that we are preparing the data below:')
            X_train = strat_train["reviews.text"]
            X_train_targetSentiment = strat_train["Sentiment"]
            X_test = strat_test["reviews.text"]
            X_test_targetSentiment = strat_test["Sentiment"]
            code3 = '''X_train = strat_train["reviews.text"]
X_train_targetSentiment = strat_train["Sentiment"]
X_test = strat_test["reviews.text"]
X_test_targetSentiment = strat_test["Sentiment"]
print(len(X_train), len(X_test))'''
            st.code(code3,language='python')
            st.write('Here we are splitting up the dataset with the train set and test set:')
            st.write(len(X_train), len(X_test))
            st.markdown('''Here we will turn content into numerical feature vectors using the Bag of Words strategy:

Assign fixed integer id to each word occurrence (integer indices to word occurrence dictionary) Xi,j where i is the integer indices, j is the word occurrence, and X is an array of words (our training set) In order to implement the Bag of Words strategy, we will use SciKit-Learn’s CountVectorizer to performs the following:

Text preprocessing: Tokenization (breaking sentences into words) Stopwords (filtering “the”, “are”, etc) Occurrence counting (builds a dictionary of features from integer indices with word occurrences) Feature Vector (converts the dictionary of text documents into a feature vector)''')
            X_train = X_train.fillna(' ')
            X_test = X_test.fillna(' ')
            X_train_targetSentiment = X_train_targetSentiment.fillna(' ')
            X_test_targetSentiment = X_test_targetSentiment.fillna(' ')
            code4 = '''# Replace "nan" with space
X_train = X_train.fillna(' ')
X_test = X_test.fillna(' ')
X_train_targetSentiment = X_train_targetSentiment.fillna(' ')
X_test_targetSentiment = X_test_targetSentiment.fillna(' ')'''
            st.code(code4,language='python')
            code5 = '''from sklearn.feature_extraction.text import CountVectorizer 
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train) 
X_train_counts.shape'''
            st.code(code5,language='python')

            from sklearn.feature_extraction.text import CountVectorizer 
            count_vect = CountVectorizer()
            X_train_counts = count_vect.fit_transform(X_train) 
            X_train_counts.shape


            st.markdown('''Here we have 22665 training samples and 9668 distinct words in our training sample.

Also, with longer documents, we typically see higher average count values on words that carry very little meaning, this will overshadow shorter documents that have lower average counts with same frequencies, as a result, we will use TfidfTransformer to reduce this redundancy:

Term Frequencies (Tf) divides number of occurrences for each word by total number of words Term Frequencies times Inverse Document Frequency (Tfidf) downscales the weights of each word (assigns less value to unimportant stop words ie. “the”, “are”, etc)''')
            from sklearn.feature_extraction.text import TfidfTransformer
            tfidf_transformer = TfidfTransformer(use_idf=False)
            X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
            code6 = '''from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(use_idf=False)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)'''
            st.code(code6,language='python')
            X_train_tfidf.shape
            st.markdown('''Building a Pipeline from the Extracted Features

We will use Multinominal Naive Bayes as our Classifier

Multinominal Niave Bayes is most suitable for word counts where data are typically represented as word vector counts (number of times outcome number Xi,j is observed over the n trials), while also ignoring non-occurrences of a feature i Naive Bayes is a simplified version of Bayes Theorem, where all features are assumed conditioned independent to each other (the classifiers), P(x|y) where x is the feature and y is the classifier''')

            code7 = '''from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
clf_multiNB_pipe = Pipeline([("vect", CountVectorizer()), 
                             ("tfidf", TfidfTransformer()),
                             ("clf_nominalNB", MultinomialNB())])
clf_multiNB_pipe.fit(X_train, X_train_targetSentiment)'''
            st.code(code7,language='python')
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline
            clf_multiNB_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()),("clf_nominalNB", MultinomialNB())])
            clf_multiNB_pipe.fit(X_train, X_train_targetSentiment)
            import numpy as np
            predictedMultiNB = clf_multiNB_pipe.predict(X_test)
            st.markdown('''Here we see that our Multinominal Naive Bayes Classifier has a 90.24% accuracy level based on the features.''')
            score = np.mean(predictedMultiNB == X_test_targetSentiment)
            st.write(score)
            st.markdown('''Next we will conduct the following:

Test other models Fine tune the best models to avoid over-fitting''')
            st.cache()
            MLModels = st.radio('Below we have used different models to find which models outputs the best score:',('Logistic Regression Classifier','Support Vector Machine Classifier','Decision Tree Classifier','Random Forest Classifier'))
            if MLModels == 'Logistic Regression Classifier':
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import Pipeline
                clf_logReg_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf_logReg", LogisticRegression())])
                clf_logReg_pipe.fit(X_train, X_train_targetSentiment)
                import numpy as np  
                predictedLogReg = clf_logReg_pipe.predict(X_test)
                st.write(np.mean(predictedLogReg == X_test_targetSentiment))
            if MLModels == 'Support Vector Machine Classifier':
                from sklearn.svm import LinearSVC
                clf_linearSVC_pipe = Pipeline([("vect", CountVectorizer()),("tfidf", TfidfTransformer()),("clf_linearSVC", LinearSVC())])
                clf_linearSVC_pipe.fit(X_train, X_train_targetSentiment)
                predictedLinearSVC = clf_linearSVC_pipe.predict(X_test)
                st.write(np.mean(predictedLinearSVC == X_test_targetSentiment))
            if MLModels == 'Decision Tree Classifier':
                from sklearn.tree import DecisionTreeClassifier
                clf_decisionTree_pipe = Pipeline([("vect", CountVectorizer()),("tfidf", TfidfTransformer()), ("clf_decisionTree", DecisionTreeClassifier())])
                clf_decisionTree_pipe.fit(X_train, X_train_targetSentiment)
                predictedDecisionTree = clf_decisionTree_pipe.predict(X_test)
                st.write(np.mean(predictedDecisionTree == X_test_targetSentiment))
            if MLModels == 'Random Forest Classifier':
                from sklearn.ensemble import RandomForestClassifier
                clf_randomForest_pipe = Pipeline([("vect", CountVectorizer()),("tfidf", TfidfTransformer()),("clf_randomForest", RandomForestClassifier())])
                clf_randomForest_pipe.fit(X_train, X_train_targetSentiment)
                predictedRandomForest = clf_randomForest_pipe.predict(X_test)
                st.write(np.mean(predictedRandomForest == X_test_targetSentiment))
            st.markdown('''Looks like all the models performed very well (>90%), and we will use the Random Forest Classifier since it has the highest accuracy level.

Now we will fine tune the Support Vector Machine model (Linear_SVC) to avoid any potential over-fitting.''')
            st.markdown('''Fine tuning the Support Vector Machine Classifier''')
            st.markdown('''Here we will run a Grid Search of the best parameters on a grid of possible values, instead of tweaking the parameters of various components of the chain (ie. use_idf in tfidftransformer) We will also run the grid search with LinearSVC classifier pipeline, parameters and cpu core maximization Then we will fit the grid search to our training data set

Finally we will test the accuracy of our final classifier (after fine-tuning) Note that Support Vector Machines is very suitable for classification by measuring extreme values between classes, to differentiate the worst case scenarios so that it can classify between Positive, Neutral and Negative correctly.''')
            code8 = '''from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],    
             'tfidf__use_idf': (True, False), 
             } 
gs_clf_LinearSVC_pipe = GridSearchCV(clf_linearSVC_pipe, parameters, n_jobs=-1)
gs_clf_LinearSVC_pipe = gs_clf_LinearSVC_pipe.fit(X_train, 
                                                  X_train_targetSentiment)
predictedGS_clf_LinearSVC_pipe = gs_clf_LinearSVC_pipe.predict(X_test) 
np.mean(predictedGS_clf_LinearSVC_pipe == X_test_targetSentiment)'''
            st.code(code8,language='python') 
            st.write(0.9590612316922534)
            st.write('''Below is the summary of the classification report:

Precision: determines how many objects selected were correct Recall: tells you how many of the objects that should have been selected were actually selected F1 score measures the weights of recall and precision (1 means precision and recall are equally important, 0 otherwise) Support is the number of occurrences of each class The results in this analysis confirms our previous data exploration analysis, where the data are very skewed to the positive reviews as shown by the lower support counts in the classification report. Also, both neutral and negative reviews has large standard deviation with small frequencies, which we would not consider significant as shown by the lower precision, recall and F1 scores in the classification report.

However, despite that Neutral and Negative results are not very strong predictors in this data set, it still shows a 96.01% accuracy level in predicting the sentiment analysis. Therefore, we are comfortable here with the skewed data set. Also, as we continue to input new dataset in the future that is more balanced, this model will then re-adjust to a more balanced classifier which will increase the accuracy level.

Note: The first row will be ignored as we previously replaced all NAN with ” “. We tried to remove this row when we first imported the raw data, but Pandas DataFrame did not like this row removed when we tried to drop all NAN (before stratifying and splitting the dataset). As a result, replacing the NAN with ” ” was the best workaround and the first row will be ignored in this analysis.

Finally, the overall result here explains that the products in this dataset are generally positively rated.

this is a result of positively skewed dataset, which is consistent with both our data exploration and sentiment analysis. Therefore, we conclude that the products in this dataset are generally positively rated, and should be kept from Amazon’s product roster.''')
            code9 = '''from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print(classification_report(X_test_targetSentiment, 
                            predictedGS_clf_LinearSVC_pipe))
print('Accuracy: {}'. format(accuracy_score(X_test_targetSentiment, 
                             predictedGS_clf_LinearSVC_pipe)))'''
            st.code(code9,language='python')
            st.code('''              precision    recall  f1-score   support

    Negative       0.84      0.78      0.81       316
     Neutral       0.88      0.47      0.62       241
    Positive       0.97      0.99      0.98      5110

    accuracy                           0.96      5667
   macro avg       0.90      0.75      0.80      5667
weighted avg       0.96      0.96      0.96      5667

Accuracy: 0.9590612316922534''')
            st.markdown('''From the analysis above in the classification report, we can see that products with lower reviews are not significant enough to predict these lower rated products are inferior. On the other hand, products that are highly rated are considered superior products, which also performs well and should continue to sell at a high level.

As a result, we need to input more data in order to consider the significance of lower rated product, in order to determine which products should be dropped from Amazon’s product roster.

The good news is that despite the skewed dataset, we were still able to build a robust Sentiment Analysis machine learning system to determine if the reviews are positive or negative. This is possible as the machine learning system was able to learn from all the positive, neutral and negative reviews, and fine tune the algorithm in order to avoid bias sentiments.

In conclusion, although we need more data to balance out the lower rated products to consider their significance, however we were still able to successfully associate positive, neutral and negative sentiments for each product in Amazon’s Catalog.''')

        if Approach == 'Approach 2':
            st.markdown('''Different Approach- using polarity function from textblob to carryout or derive negative and positive sentiment for the text(reviews)and then running the model and testing its accuracy''')
            st.markdown('''If polarity precision is important to your business, you might consider expanding your polarity categories to include:

Very positive Positive Neutral Negative Very negative This is usually referred to as fine-grained sentiment analysis, and could be used to interpret 5-star ratings in a review, for example:

Very Positive = 5 stars Very Negative = 1 star''')
            st.markdown('''Creating 'Subjectivity' and 'Polarity' Scores

Polarity is float which lies in the range of -1,1 where 1 means positive statement and -1 means a negative statement.

Subjective sentences generally refer to personal opinion, emotion or judgment whereas objective refers to factual information. Subjectivity is also a float which lies in the range of 0,1.''')

            pol = lambda x: TextBlob(x).sentiment.polarity
            sub = lambda x: TextBlob(x).sentiment.subjectivity
            strat_test['polarity'] = strat_test["reviews.text"].apply(pol)
            strat_test['subjectivity'] = strat_test["reviews.text"].apply(sub)
            strat_train['polarity'] = strat_train["reviews.text"].apply(pol)
            strat_train['subjectivity'] = strat_train["reviews.text"].apply(sub)
            df['polarity'] = df["reviews.text"].apply(pol)
            df['subjectivity'] = df["reviews.text"].apply(sub)

            st.subheader('Histogram of Polarity Score Entire Data')
            st.image('./Stream1.PNG')
            st.subheader('Histogram of Polarity Score Train Data')
            st.image('./Stream2.PNG')
            st.subheader('Histogram of Polarity Score Test Data')
            st.image('./Stream3.PNG')

            st.markdown('We Can See that Most of the Reviews have Neutral Sentiment')

            st.markdown('Below we are making a new column in the dataset to reflect the sentiment value from the reviews polarity see below. We have used Polarty function from the textblob library to derive the sentiments from the text(reviews) and We have sucessfully map the sentiment to each product and now we will be testing it for our model accuracy')
            def New_sentiments(polarity):
                if (polarity > 0) :
                    return "Positive"
                elif polarity == 0:
                    return "Neutral"
                elif polarity < 0:
                    return "Negative"
            strat_train["NEW_Sentiment"] = strat_train["polarity"].apply(New_sentiments)
            strat_test["NEW_Sentiment"] = strat_test["polarity"].apply(New_sentiments)
            st.write(strat_train[['NEW_Sentiment','polarity']].head())

            X_train = strat_train["reviews.text"]
            X_train_targetSentiment = strat_train["NEW_Sentiment"]
            X_test = strat_test["reviews.text"]
            X_test_targetSentiment = strat_test['NEW_Sentiment']

            X_train = X_train.fillna(' ')
            X_test = X_test.fillna(' ')
            X_train_targetSentiment = X_train_targetSentiment.fillna(' ')
            X_test_targetSentiment = X_test_targetSentiment.fillna(' ')

            from sklearn.feature_extraction.text import CountVectorizer 
            count_vect = CountVectorizer()
            X_train_counts = count_vect.fit_transform(X_train) 
            from sklearn.feature_extraction.text import TfidfTransformer
            tfidf_transformer = TfidfTransformer(use_idf=False)
            X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline
            clf_multiNB_pipe = Pipeline([("vect", CountVectorizer()), 
                             ("tfidf", TfidfTransformer()),
                             ("clf_nominalNB", MultinomialNB())])
            clf_multiNB_pipe.fit(X_train, X_train_targetSentiment)

            st.cache()
            MLModels2 = st.radio('Below we have used different models to find which models outputs the best score:',('Logistic Regression Classifier','Support Vector Machine Classifier','Decision Tree Classifier','Random Forest Classifier'))
            if MLModels2 == 'Logistic Regression Classifier':
                import numpy as np  
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import Pipeline
                clf_logReg_pipe = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf_logReg", LogisticRegression())])
                clf_logReg_pipe.fit(X_train, X_train_targetSentiment)
                predictedLogReg = clf_logReg_pipe.predict(X_test)
                st.write(np.mean(predictedLogReg == X_test_targetSentiment))
            if MLModels2 == 'Support Vector Machine Classifier':
                import numpy as np  
                from sklearn.svm import LinearSVC
                clf_linearSVC_pipe = Pipeline([("vect", CountVectorizer()),("tfidf", TfidfTransformer()),("clf_linearSVC", LinearSVC())])
                clf_linearSVC_pipe.fit(X_train, X_train_targetSentiment)
                predictedLinearSVC = clf_linearSVC_pipe.predict(X_test)
                st.write(np.mean(predictedLinearSVC == X_test_targetSentiment))
            if MLModels2 == 'Decision Tree Classifier':
                import numpy as np  
                from sklearn.tree import DecisionTreeClassifier
                clf_decisionTree_pipe = Pipeline([("vect", CountVectorizer()),("tfidf", TfidfTransformer()), ("clf_decisionTree", DecisionTreeClassifier())])
                clf_decisionTree_pipe.fit(X_train, X_train_targetSentiment)
                predictedDecisionTree = clf_decisionTree_pipe.predict(X_test)
                st.write(np.mean(predictedDecisionTree == X_test_targetSentiment))
            if MLModels2 == 'Random Forest Classifier':
                import numpy as np  
                from sklearn.ensemble import RandomForestClassifier
                clf_randomForest_pipe = Pipeline([("vect", CountVectorizer()),("tfidf", TfidfTransformer()),("clf_randomForest", RandomForestClassifier())])
                clf_randomForest_pipe.fit(X_train, X_train_targetSentiment)
                predictedRandomForest = clf_randomForest_pipe.predict(X_test)
                st.write(np.mean(predictedRandomForest == X_test_targetSentiment))
            st.markdown('''With this approach 2 we can say that our model not only have predicted the correct sentiment for the different amazon products but also has increased the accuracy level to 97.35%.

Thus based on the analysis and after comparing two approaches we are rejecting approch 1 and accepting approach 2''')

if __name__ == '__main__':
    	main()