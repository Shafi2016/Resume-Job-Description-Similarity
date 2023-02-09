import PyPDF2
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
st.set_option('deprecation.showPyplotGlobalUse', False)
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def remove_stop_words(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])

# Load the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def show_wordcloud(text, title):
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=800).generate(text)
    #st.write(title)
    st.image(wordcloud.to_array() , use_column_width=True)

def compute_cosine_similarity(resume_text, job_description_text):
    # Compute embeddings for both lists
    embeddings1 = model.encode(resume_text, convert_to_tensor=True)
    embeddings2 = model.encode(job_description_text, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    # Return the average cosine similarity
    return cosine_scores.mean()

def show_bar_chart(text, title):
    
    vectorizer = TfidfVectorizer()
    
    response = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = zip(feature_names, np.asarray(response.sum(axis=0)).ravel()) 
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    words = [x[0] for x in sorted_scores[:15]]
    values = [x[1] for x in sorted_scores[:15]]
    y_pos = np.arange(len(words))

    plt.barh(y_pos, values, align='center', alpha=0.5)
    plt.yticks(y_pos, words)
    plt.xlabel('TF-IDF Score')
    #plt.title(title)

    #st.write(title)
    st.pyplot()
def main():
    st.title("Resume-Job Description Similarity")
    st.subheader('Increase your chances of getting shortlisted')
    st.write("Using the Sentence-BERT (SBERT) model, calculate the similarity between any two documents. It presents a word cloud and bar chart of significant words.")

    # Add a sidebar to the app
    st.sidebar.title("Upload Files")

    # Put the file uploader widgets in the sidebar
    st.sidebar.write("Upload your Resume PDF:")
    resume_pdf = st.sidebar.file_uploader("", type=["pdf"])

    st.sidebar.write("Upload Job Description PDF:")
    job_description_pdf = st.sidebar.file_uploader("Job Description PDF", type=["pdf"], key='job_description_pdf')

    if resume_pdf and job_description_pdf:
        resume_text = extract_text_from_pdf(resume_pdf)
        job_description_text = extract_text_from_pdf(job_description_pdf)
        resume_text = remove_stop_words(resume_text)
        job_description_text = remove_stop_words(job_description_text)
        
        similarity_score = compute_cosine_similarity(resume_text, job_description_text)
        st.write("**Cosine Similarity Score:**", bold=True)
        st.write("<div style='background-color: PowderBlue; border: 2px solid blue; padding: 12px; font-size: smaller;'>{:.4f}</div>".format(similarity_score), unsafe_allow_html=True)
        st.write("The cosine similarity score value ranges from 0–1. 0 means no similarity, where as 1 means that both the items are 100% similar", box=True)

        # Use the st.tabs method to create tabs
        tabs = ["Word Clouds", "Bar Charts"]
        selected_tab = st.selectbox("Select a Tab", tabs)

        if selected_tab == "Word Clouds":
            st.header("Resume Word Cloud")
            show_wordcloud(resume_text, "Resume Word Cloud")
            st.header("Job Description Word Cloud")
            show_wordcloud(job_description_text, "Job Description Word Cloud")
        if selected_tab == "Bar Charts":
            st.header("Resume Bar Chart")
            show_bar_chart(resume_text, "Resume Bar Chart")
            st.header("Job Description Bar Chart")
            show_bar_chart(job_description_text, "Job Description Bar Chart")

        
if __name__ == '__main__':
    main()

