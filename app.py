import streamlit as st
from simpletransformers.question_answering import QuestionAnsweringModel
import torch

# Check if CUDA is available
use_cuda = torch.cuda.is_available()

model_path = "best_model" 
model = QuestionAnsweringModel("roberta", model_path, use_cuda=use_cuda)

# Streamlit UI
st.title("Question Answering System")

context = st.text_area("Enter Context:", height=200)
question = st.text_input("Enter Question:")

if st.button("Answer"):
    if context and question:
        answers, probabilities = model.predict([{"context": context, "qas": [{"question": question, "id": "001"}]}])

        # Extract the answer from the list and remove the index
        answer = answers[0]['answer'][0]  
        if answer:
            st.write("**Answer :**\n", answer)
        else:
            st.write("**Answer :** \n No answer found.")
    else:
        st.warning("Please enter both context and question.")