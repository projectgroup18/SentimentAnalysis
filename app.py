import torch.nn as nn
import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from transformers import BertForSequenceClassification, BertConfig


# We can use a cased and uncased version of BERT and tokenizer. I am using cased version.
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

# Let's load a pre-trained BertTokenizer
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

sample_txt = "we love you"

encoding = tokenizer.encode_plus(
    sample_txt,
    padding='max_length', # Pad sentence to max length
    truncation=True,  #Truncate sentence to max length
    max_length=32,
    add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=False,
    return_attention_mask=True, # Return attention mask
    return_tensors='pt',  # Return torch objects
    )

encoding.keys()

# BERT works with fixed-length sequences. We'll use a simple strategy to choose the max length.
# Let's store the token length of each review..

token_lens = []

MAX_LEN = 150

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,return_dict=False)

last_hidden_state, pooled_output = bert_model(
  input_ids=encoding['input_ids'],
  attention_mask=encoding['attention_mask']
)
# The last_hidden_state is a sequence of hidden states of the last layer of the model.
# Obtaining the pooled_output is done by applying the BertPooler on last_hidden_state.
last_hidden_state.shape

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,return_dict=False)
        # dropout layer for some regularization
        self.drop = nn.Dropout(p=0.3)
        # A fully-connected layer for our output
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        last_hidden_state,pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


class_names = ['negative', 'neutral', 'positive']

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentimentClassifier(len(class_names))
model.load_state_dict(torch.load('best_model_state.bin',map_location=torch.device('cpu')))
model = model.to(device)


def prediction_on_raw_data(text_input):
    encoded_review = tokenizer.encode_plus(
        text_input,
        max_length=32,
        add_special_tokens=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, prediction = torch.max(outputs, dim=1)
        probability = torch.softmax(outputs, dim=1)[0]
        
    return class_names[prediction], probability
        
def main():
    st.title("Sentiment Analysis on Text Data")
    config = BertConfig.from_pretrained('bert-base-cased')


    # Define the BERT model and tokenizer
    bert_model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name)
    

    
    # Load the trained model state dictionary
    model_path = './best_model_state.bin'  # Update with your model path
    
    # Load the trained model state dictionary
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        # If running on a CPU-only machine, map the model parameters to CPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # User input for text
    text_input = st.text_area("Enter your text here:")
    
    if st.button("Analyze"):
        if text_input:
            # Predict sentiment
            sentiment, probability = prediction_on_raw_data(text_input)
            st.write(f"Sentiment: {sentiment}")
            st.write(f"probability: {probability[class_names.index(sentiment)].item()*100:.2f}%")
            
            

if __name__ == "__main__":
    main()
