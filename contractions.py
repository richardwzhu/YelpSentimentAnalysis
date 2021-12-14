import re
def contractions(s):
 s = re.sub(r"won’t", "will not",s)
 s = re.sub(r"would’t", "would not",s)
 s = re.sub(r"could’t", "could not",s)
 s = re.sub(r"\’d", " would",s)
 s = re.sub(r"can\’t", "can not",s)
 s = re.sub(r"n\’t", " not", s)
 s = re.sub(r"\’re", " are", s)
 s = re.sub(r"\’s", " is", s)
 s = re.sub(r"\’ll", " will", s)
 s = re.sub(r"\’t", " not", s)
 s = re.sub(r"\’ve", " have", s)
 s = re.sub(r"\’m", " am", s)
 return s