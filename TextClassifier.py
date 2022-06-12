import numpy as np
import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.datasets import load_files
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

class TextClassifier:
    def __init__(self):
        self.data = []
        self.preprocessed_data = []
        self.target = []
        self.bow = []
        self.train_data = []
        self.test_data = []
        self.train_target = []
        self.test_correct_target = []
        self.test_output_target = []
        self.classifier = []

    def PreprocessData(self, path):
        self.ReadDataFromFolder(path)
        self.TextPreprocessing(self.data)
        self.ConvertToBOW()

    def ReadDataFromFolder(self, path):
        #load_files divide datasets into data and targets and treat each subdirectory of given folder as a separate category
        movie_data = load_files(r"mlarr_text")
        self.data, self.target = movie_data.data, movie_data.target

    def TextPreprocessing(self, data):
        lemmatizer = WordNetLemmatizer()
        self.preprocessed_data = []
        for doc in range(0, len(data)):
            # Remove all non word characters and replace them with a space
            data_after_regex = re.sub(r'\W', ' ', str(data[doc]))

            # after removing special characters we might get ''dog's'' converted to ''dog s'' so we want to remvoe single leter characters
            data_after_regex = re.sub(r'\s+[a-zA-Z]\s+', ' ', data_after_regex)

            # remove double spaces
            data_after_regex = re.sub(r'\s+', ' ', data_after_regex, flags=re.I)

            # we don't want ''Dog'' and ''dog'' to be treated as different word 
            data_after_regex = data_after_regex.lower()

            # convert plural to singular and other similar word for example ''dogs'' into ''dog''
            data_after_regex = data_after_regex.split()

            data_after_regex = [lemmatizer.lemmatize(word) for word in data_after_regex]
            data_after_regex = ' '.join(data_after_regex)

            self.preprocessed_data.append(data_after_regex)


    def ConvertToBOW(self):
        vectorizer = CountVectorizer(max_features=2000, min_df=5, max_df=0.8, stop_words=stopwords.words('english'))
        bow = vectorizer.fit_transform(self.preprocessed_data).toarray()
        #need to be converted using Term Frequency Inverse Document Frequency
        tfidfconverter = TfidfTransformer()
        self.bow = tfidfconverter.fit_transform(bow).toarray()

    def RunTraining(self):
        self.SplitDataToTrainAndTest()
        self.TrainModel()

    def SplitDataToTrainAndTest(self):
        self.train_data, self.test_data, self.train_target, self.test_correct_target = train_test_split(self.bow, self.target, test_size=0.3, random_state=0)

    def TrainModel(self):
        self.classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
        self.classifier.fit(self.train_data, self.train_target) 
        self.test_output_target = self.classifier.predict(self.test_data)
    
    def EvaluateModel(self):
        print(confusion_matrix(self.test_correct_target, self.test_output_target))
        print(classification_report(self.test_correct_target, self.test_output_target))
        print(accuracy_score(self.test_correct_target, self.test_output_target))

    def PredictSingleFile(self, path):
        with open(path, 'r') as file:
            file_content = file.read().replace('\n', ' ')
            self.TextPreprocessing(file_content)
        vectorizer = CountVectorizer(max_features=2000, min_df=1, max_df=1, stop_words=stopwords.words('english'))
        text = vectorizer.transform(self.preprocessed_data).toarray()
        tfidfconverter = TfidfTransformer()
        text = tfidfconverter.transform(text).toarray()
        print({self.classifier.predict(text)[0]})


if __name__ == "__main__":
    text_classifier = TextClassifier()
    text_classifier.PreprocessData("mlarr_text")
    text_classifier.RunTraining()
    text_classifier.EvaluateModel()
    text_classifier.PredictSingleFile("mlarr_text/business/b_406.txt")