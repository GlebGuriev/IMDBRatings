from django.shortcuts import render
from django.http import HttpResponse

from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
from keras.models import load_model

from numpy import ceil
import spacy, re
nlp = spacy.load("en_core_web_sm")

with open('imdbratings/model/tokenizer.json', 'r') as f:
    tokenizer_json = f.read()

tokenizer = tokenizer_from_json(tokenizer_json)
rating_model = load_model('imdbratings/model/imdbratings.h5')
stopwords = nlp.Defaults.stop_words

# Create your views here.
def index(request):
    return render(request, "imdbratings/index.html")

def model(request, review):
    def get_rating(review):
        emoji_pattern = re.compile(pattern="["
                                               u"\U0001F600-\U0001F64F"  # emoticons
                                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                               "]+", flags=re.UNICODE)
        unemoji = emoji_pattern.sub(r'', review)

        tag_pattern = re.compile('<.*?>')
        untag = tag_pattern.sub(r'', unemoji)
        lowercase = untag.lower()

        no_stopwords = ' '.join([word for word in lowercase.split() if word not in stopwords])
        lemmas = ' '.join([word.lemma_ for word in nlp(no_stopwords)])

        tokenized = tokenizer.texts_to_sequences([lemmas])
        padded = pad_sequences(tokenized, maxlen=548, padding='post')
        prediction = rating_model.predict(padded)

        result = f'{int(ceil(prediction * 10))}/10, положительная оценка' if prediction > 0.5 else \
            f'{int(ceil(prediction * 10))}/10, отрицательная оценка'

        return result

    return HttpResponse(get_rating(review))
