import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from morpheus_hf.morpheus import MorpheusHuggingfaceNLI, MorpheusHuggingfaceQA, MorpheusHuggingfaceSummarization

model_name = st.selectbox(
     'Select a qa model',
     ('deepset/roberta-base-squad2', 'deepset/bert-base-cased-squad2'))

model_name = 'deepset/bert-base-cased-squad2'

def load_qa_model():
    model = pipeline('question-answering', model=model_name, tokenizer=model_name)
    return model

qa = load_qa_model()



st.title("Ask Questions about your Text")
sentence = st.text_area('Please paste your article :', height=30)
question = st.text_input("Questions from this article?")
button = st.button("Get me Answers")
with st.spinner("Discovering Answers.."):
    if button and sentence:
        result = qa(question=question, context=sentence)
        st.write(result['answer'], result['score'], result['start'], result['end'])
        
        test_morph_qa = MorpheusHuggingfaceQA(model_name)
        context = sentence
        q_dict = {"question": question , "id": "56ddde6b9a695914005b9628", "answers": [{"text": result['answer'], "answer_start": result['start']}], "is_impossible": False}

        text = test_morph_qa.morph(q_dict, context)
        st.write(text[0])
        
        answersEdited = qa(question=text[0], context=sentence)
        st.write(answersEdited['answer'])
