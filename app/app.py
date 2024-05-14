import joblib
import keras
from keras.utils import pad_sequences
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import preprocess_input
import numpy as np
from fastapi import FastAPI, File, UploadFile
import os
from PIL import Image
import shutil
import google.generativeai as genai
from mangum import Mangum

genai.configure(api_key="AIzaSyCFF-rfVi8KH1gNHuX93n6j9BF7468D1Mw")

# Set up the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 8192,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

aimodel = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)




err_str = "#######"

model = load_model('app/lstm_for_primary_headline_generation.h5')
tokenizer = joblib.load("app/tokenizer_for_garbage.pkl")
mytokenizer = joblib.load("app/tokenizer_for_short.pkl")
short_headline_model = load_model('app/lstm_for_short_headline_description.h5')


vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences([sequence], max_length)

        yhat = model.predict([image, sequence], verbose=0)

        yhat = np.argmax(yhat)

        word = idx_to_word(yhat, tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == 'endseq':
            break

    return in_text

def purify(arr):
  ans_str = ""
  z = []
  for i in range(len(arr)-1):
    for j in range(i+1,len(arr)):
      if(arr[i]==arr[j]):
        arr[j] = err_str
      else:
        break
  for i in range(len(arr)):
    if(arr[i] != err_str):
      z.append(arr[i])
  for i in range(len(z)):
    ans_str+=z[i]
    ans_str+= " "
  return ans_str


def generate_headlines_from_images(img_store):
    arr_ans = []
    for image_path in img_store:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
        img_array = preprocess_input(img_array)

        # Assuming vgg_model, model, tokenizer, predict_caption, and purify functions are defined

        feature = vgg_model.predict(img_array, verbose=0)
        caption = predict_caption(model, feature, tokenizer, 17)

        # Adjust indices as per your requirement
        start = 8
        end = len(caption) - 7
        ans = ""
        for i in range(start, end):
            ans += caption[i]
        arr = ans.split(' ')
        arr.pop(0)
        err_str = "#######"
        ans = purify(arr)
        arr_ans.append(ans)

    return arr_ans

def short_headline(arr_ans):
    input_text = arr_ans[0]
    predict_next_words= 100
    max_sequence_len = 77
    for _ in range(predict_next_words):
        token_list = mytokenizer.texts_to_sequences([input_text])[0]
        # print(token_list)
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(short_headline_model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in mytokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        input_text += " " + output_word

    return input_text

# ans = generate_headlines_from_images(img_store)
# Text_input_1 = "I have a an array and want to make news headline from words in it, output just a string and nothing else of news headline just give me headline, no code no explanation"
# Text_input_2 = ans[0]
# response = aimodel.generate_content([Text_input_1, Text_input_2], stream=True)
# response.resolve()
# headline = response.text
# print(headline, "\n")




# news = short_headline(ans)
# News_input_1 = "I have a string outputted from a model with words, make a structured short news paragraph from it and output just that paragraph, no explanation and anything, just paragraph interpreted from the give input string"
# News_input_2 = news
# news_response = aimodel.generate_content([News_input_1, News_input_2], stream=True)
# news_response.resolve()
# article = news_response.text
# print(article)

app = FastAPI()

handler = Mangum(app)


@app.get('/')
async def root():
    return {"Hello" : "world"}


@app.post('/predict')
async def make_news(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        with open("temp_image.jpeg", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the uploaded file
        img_store = ["temp_image.jpeg"]
        # headlines = generate_headlines_from_images(img_store)
        ans = generate_headlines_from_images(img_store)
        Text_input_1 = "I have a an array and want to make news headline from words in it, output just a string and nothing else of news headline just give me headline, no code no explanation"
        Text_input_2 = ans[0]
        response = aimodel.generate_content([Text_input_1, Text_input_2], stream=True)
        response.resolve()
        headline = response.text
        # print(headline, "\n")

        # news = short_headline(ans)
        news = short_headline(ans)
        News_input_1 = "I have a string outputted from a model with words, make a structured short news paragraph from it and output just that paragraph, no explanation and anything, just paragraph interpreted from the give input string"
        News_input_2 = news
        news_response = aimodel.generate_content([News_input_1, News_input_2], stream=True)
        news_response.resolve()
        article = news_response.text
        # print(article)
        return {"headlines": headline, "NEWS" : article}

    finally:
        # Clean up the temporary file
        if os.path.exists("temp_image.jpg"):
            os.remove("temp_image.jpg")
