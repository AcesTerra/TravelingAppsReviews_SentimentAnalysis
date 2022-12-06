import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from torch import nn, optim

class_names = ['negative', 'neutral', 'positive']
PRE_TRAINED_MODEL_NAME = 'AcesTerra/PlayStore_TravelgApps_SentimentAnalisis'
MAX_LEN = 160

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = model = SentimentClassifier(len(class_names))
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Enter text to analyze')
button = st.button("Analyze")

#d = {
    
#  0:'Negative',
#  1:'Neutral',
#  2:'Positive'

#}

if user_input and button :
    encoded_review = tokenizer.encode_plus(
      user_input,
      max_length=MAX_LEN,
      add_special_tokens=True,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    st.write("Review text:", user_input)
    st.write("Sentiment:", class_names[prediction])
    #test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    #output = model(**test_sample)
    #st.write("Logits: ",output.logits)
    #y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    #st.write("Prediction: ",d[y_pred[0]])
