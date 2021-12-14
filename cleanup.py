import numpy as np
import pandas as pd

# Cleaning texts
import nltk
import re
from stop_list import closed_class_stop_words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

cleanup = open("Restaurant_Reviews3.txt", "w")
cleanupDev = open("dev3.txt", "w")
cleanupTest = open("test3.txt", "w")
cleanupTrain = open("train3.txt", "w")
with open("Restaurant_Reviews.txt", 'r') as file, open("dev.txt", 'r') as fileDev, open("test.txt", 'r') as fileTest,  open("train.txt", 'r') as fileTrain:
    while (line := file.readline().rstrip()):
        # Cleaning special character from the reviews
        num = line.split("\t")[1]
    
        line = re.sub(pattern='[^a-zA-Z]',repl=' ', string=line)

         # Converting the entire review into lower case
        line = line.lower()

        # Tokenizing the review by words
        line_words = line.split()

        # Removing the stop words
        line_words = [word for word in line_words if not word in closed_class_stop_words]

        # Stemming the words
        ps = PorterStemmer()
        line = [ps.stem(word) for word in line_words]

        # Joining the stemmed words
        line = ' '.join(line)

        cleanup.write(line + "\t" + num + "\n")   
# Dev
    while (line := fileDev.readline().rstrip()):
        num = line.split("\t")[1]
        # Cleaning special character from the reviews
        line = re.sub(pattern='[^a-zA-Z]',repl=' ', string=line)

         # Converting the entire review into lower case
        line = line.lower()

        # Tokenizing the review by words
        line_words = line.split()

        # Removing the stop words
        line_words = [word for word in line_words if not word in closed_class_stop_words]

        # Stemming the words
        ps = PorterStemmer()
        line = [ps.stem(word) for word in line_words]

        # Joining the stemmed words
        line = ' '.join(line) 

        cleanupDev.write(line + "\t" + num + "\n")   
# Test 
    while (line := fileTest.readline().rstrip()):
        num = line.split("\t")[1]
        # Cleaning special character from the reviews
        line = re.sub(pattern='[^a-zA-Z]',repl=' ', string=line)

         # Converting the entire review into lower case
        line = line.lower()

        # Tokenizing the review by words
        line_words = line.split()

        # Removing the stop words
        line_words = [word for word in line_words if not word in closed_class_stop_words]

        # Stemming the words
        ps = PorterStemmer()
        line = [ps.stem(word) for word in line_words]

        # Joining the stemmed words
        line = ' '.join(line)

        cleanupTest.write(line + "\t" + num + "\n")     
# Train
    while (line := fileTrain.readline().rstrip()):
        num = line.split("\t")[1]
        # Cleaning special character from the reviews
        line = re.sub(pattern='[^a-zA-Z]',repl=' ', string=line)

         # Converting the entire review into lower case
        line = line.lower()

        # Tokenizing the review by words
        line_words = line.split()

        # Removing the stop words
        line_words = [word for word in line_words if not word in closed_class_stop_words]

        # Stemming the words
        ps = PorterStemmer()
        line = [ps.stem(word) for word in line_words]

        # Joining the stemmed words
        line = ' '.join(line)

        cleanupTrain.write(line + "\t" + num + "\n")    
cleanup.close()
cleanupDev.close()
cleanupTest.close()
cleanupTrain.close()
