import streamlit as st
from simpletransformers.question_answering import QuestionAnsweringModel
import torch

# Check if CUDA is available
use_cuda = torch.cuda.is_available()

model_path = "best_model"
model = QuestionAnsweringModel("roberta", model_path, use_cuda=use_cuda)

# Streamlit UI
st.title("Question Answering System")
st.write("This project is a final submission for a Natural Language Processing (NLP) course. It implements a Question Answering System using the RoBERTa pretrained model and the SQuAD dataset.")
st.write("---")
st.write("Instructions:")
st.write("1. Enter the context in the text area.")
st.write("2. Enter the question in the text box.")
st.write("3. Click the 'Answer' button to get the answer to the question.")

context = st.text_area(
    "Enter Context:", placeholder="Add the text you want to use as context here", height=200)
question = st.text_input(
    "Enter Question:", placeholder="Type your question about the context")

if st.button("Answer"):
    if context and question:
        answers, probabilities = model.predict(
            [{"context": context, "qas": [{"question": question, "id": "001"}]}])

        # Extract the answer from the list and remove the index
        answer = answers[0]['answer'][0]
        probability = probabilities[0]['probability'][0]
        if answer and probability > 0.75:
            st.write("**Answer :**", answer)
            st.write("**Probability :**", probability)
        else:
            st.write("**Answer :** No answer found.")
            st.write("**Probability :** < 0.75")
    else:
        st.warning("Please enter both context and question.")
