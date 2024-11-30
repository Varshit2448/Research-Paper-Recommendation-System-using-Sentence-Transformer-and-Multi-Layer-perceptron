import streamlit as st
import torch
from sentence_transformers import SentenceTransformer , util
import pickle
from keras.layers import TextVectorization
import keras
import numpy as np
import tensorflow as tf

embeddings = pickle.load(open("Models/embeddings.pkl" , 'rb'))
tences = pickle.load(open("Models/sentences.pkl" , 'rb'))
rec_model = pickle.load(open("Models/Trained_model.pkl" , 'rb'))



# Load the configuration
with open('Models/text_vectorizer_config.pkl', 'rb') as f:
    loaded_config = pickle.load(f)

# Create a new TextVectorization layer from the config
loaded_text_vectorizer = TextVectorization.from_config(loaded_config)

# Load the vocabulary and set it in the new layer
with open('Models/text_vectorizer_weights.pkl', 'rb') as f:
    loaded_vocab = pickle.load(f)
    loaded_text_vectorizer.set_vocabulary(loaded_vocab)

print("TextVectorization layer loaded successfully with the vocabulary")
loaded_model = keras.models.load_model("Models/model_1")

def recommendation(input_paper):
    input_embedding = rec_model.encode(input_paper)
    cosine_scores = util.cos_sim(embeddings , input_embedding)
    #k = min(5, cosine_scores.size(0))
    top_similar_papers = torch.topk(cosine_scores , dim = 0 , k = 8 , sorted = True)

    paper_list = set()
    for i in top_similar_papers.indices:
        paper_list.add(tences[i.item()])

    return paper_list

def invert_multi_hot(encoded_labels):
    """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
    try:
        hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
        if hot_indices.size == 0:
            print("Warning: No active indices found in encoded labels.")
        return np.take(loaded_vocab, hot_indices)
    except IndexError as e:
        print(f"Index error during label lookup: {e}")
        return ['[UNK]']  # Return a placeholder for unexpected issues.

def predict_category(abstract, model, vectorizer, label_lookup):
    try:
        # Ensure the input is a list containing the abstract string
        preprocessed_abstract = [abstract]
        print("Step 1 - Shape of abstract:", np.shape(preprocessed_abstract))

        # Pass the input directly to the model without calling vectorizer outside of it
        preprocessed_abstract = tf.convert_to_tensor(preprocessed_abstract)
        print("Step 2 - Shape of preprocessed_abstract:", preprocessed_abstract.shape)

        # Make predictions using the loaded model (model should handle vectorization)
        predictions = model.predict(preprocessed_abstract)
        print("Shape of predictions:", predictions.shape)

        # Convert predictions to human-readable labels
        predicted_labels = label_lookup(np.round(predictions).astype(int)[0])

        return predicted_labels
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

st.title("The Library of Pergamum")
input_paper = st.text_input("Enter Your paper's Title")
new_abstract = st.text_area("Enter Your paper's abstract")
if st.button("Recommend"):
    recommend_paper = recommendation(input_paper)
    st.subheader("Recommended Papers")
    st.write(recommend_paper)

    predicted_categories = predict_category(new_abstract, loaded_model, loaded_text_vectorizer, invert_multi_hot)