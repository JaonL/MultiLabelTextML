from flask import Flask, jsonify, request
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import load_model
import pickle

from toxic_helper import preprocess_text, maxlen

app = Flask(__name__)

model = load_model('model_toxic.h5')

# loading tokenizer
with open('tokenizer_toxic.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        userjson = request.get_json()
        teststring = [preprocess_text(userjson['text'])]
        teststring = tokenizer.texts_to_sequences(teststring)
        teststring = pad_sequences(teststring, padding='post', maxlen=maxlen)

        res = model.predict(np.expand_dims(teststring[0], 0))
        tagNames = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        print(tagNames)
        print(res)
        predictedTags = []
        scores = []

        for i, item in enumerate(res[0]):
            if item >= 0.5:
                predictedTags.append(tagNames[i])
                scores.append(str(res[0][i]))

        # indexes = sorted(range(len(res[0])), key=lambda i: res[0][i])[-3:]  # finds top 3 possible tags
        # for i in indexes:
        #     predictedTags.append(tagNames[i])
        #     scores.append(str(res[0][i]))

        return jsonify({'tags': predictedTags, 'scores': scores})
    else:
        return 'This is the ML API'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
