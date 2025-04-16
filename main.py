import streamlit as st 
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image 
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

@st.cache_resource
def load_plant_model():
    # Load the pre-trained model
    model = load_model("plant_classification_modelv2.h5")
    return model

# Load the model
model = load_plant_model()

# Define class labels
class_labels = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 
                'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 
                'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 
                'Sugar beet']

def model_prediction(image):
    if isinstance(image, str):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(image)
    
    # Resize to match the input size expected by EfficientNet (224x224 is common)
    img = cv2.resize(img, (224, 224))
    
    # Preprocess input for EfficientNet
    img = preprocess_input(img)
    
    # Add batch dimension
    input_arr = np.expand_dims(img, axis=0)
    
    # Make prediction
    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions)
    predicted_prob = predictions[0][predicted_index] * 100
    
    return class_labels[predicted_index], predicted_prob

st.sidebar.title("Classification des Espèces de Plantes")
app_mode = st.sidebar.selectbox("Sélectionner la Page", ["Accueil", "À Propos du Projet", "Prédiction"])

if app_mode == "Accueil":
    st.header(" Classification d'images de semis de cultures et de mauvaises herbes")
    st.subheader("Classification des plantules en douze catégories d'espèces:")
    st.write("Cette application aide à identifier différentes espèces de plantules à l'aide d'un modèle d'apprentissage profond.")
    st.image("dataset-cover.jpg")

elif app_mode == "À Propos du Projet":
    st.header("À Propos du Projet")
    st.subheader("Aperçu du Jeu de Données")
    st.text("""Le jeu de données est composé d'images de plantules à différents stades de croissance.
Chaque image appartient à l'une des douze espèces de plantes:""")
    st.code(", ".join(class_labels))
    
    st.subheader("Source du Jeu de Données")
    st.write("Le jeu de données provient de Kaggle: V2 Plant Seedlings Dataset")
    st.write("https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset")
    
    st.subheader("Objectif")
    st.text("""L'objectif principal est de classifier avec précision les plantules dans leurs espèces respectives.
Cela aide à l'identification précoce des cultures par rapport aux mauvaises herbes 
dans les contextes agricoles.""")
    
    st.subheader("Informations sur le Modèle")
    st.text("Le modèle est basé sur EfficientNet avec des couches de classification personnalisées.")
    st.code("""
1. Prétraitement des Données:
   - Les images RGB sont redimensionnées à 224x224 pixels.
   - Les images sont normalisées à l'aide du prétraitement EfficientNet.

2. Architecture du Modèle:
   - Utilise un EfficientNet pré-entraîné comme modèle de base.
   - Couches denses supplémentaires: 
     - Dense(256, activation='relu')
     - Dense(128, activation='relu')
     - Dropout(0.4)
   - Couche de sortie avec 12 unités et activation softmax pour la classification des espèces de plantes.

3. Interface de Prédiction:
   - Construite à l'aide de Streamlit.
   - Permet de télécharger des images personnalisées pour la prédiction.
   - Affiche la prédiction du modèle avec le niveau de confiance.
""")

elif app_mode == "Prédiction":
    st.header("Prédiction d'Espèce de Plante")
    st.subheader("Sélectionnez une classe de plante ou téléchargez une image personnalisée:")
    
    selected_option = st.radio(
        "Choisissez une option:",
        ["Télécharger une Image Personnalisée", "Parcourir les Images d'Exemple"]
    )
    
    if selected_option == "Télécharger une Image Personnalisée":
        uploaded_image = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None:
            img = Image.open(uploaded_image).convert("RGB")
            st.image(img, caption="Image Téléchargée", width=300)

            if st.button("Prédire"):
                predicted_label, predicted_prob = model_prediction(img)
                st.success(f"Le modèle prédit: {predicted_label}")
                st.progress(predicted_prob / 100)
                st.write(f"Confiance: {predicted_prob:.2f}%")
    
    else:  # Parcourir les Images d'Exemple
        st.subheader("Sélectionnez une espèce de plante pour voir des images d'exemple:")
        selected_class = st.selectbox("Choisissez une espèce de plante", class_labels)
        test_folder = f"data/test/{selected_class.lower().replace(' ', '_')}"
        
        # Check if the folder exists
        if not os.path.exists(test_folder):
            st.warning(f"Dossier d'exemples pour '{selected_class}' introuvable. Assurez-vous que votre jeu de données est organisé comme prévu.")
            st.info("Structure de dossier attendue: data/test/[dossier_espèce_plante]")
        else:
            # Get list of image files
            if "current_class" not in st.session_state or st.session_state.current_class != selected_class:
                st.session_state.current_class = selected_class
                image_files = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    st.session_state.selected_images = random.sample(image_files, min(6, len(image_files)))
                else:
                    st.session_state.selected_images = []

            if not st.session_state.selected_images:
                st.error(f"Aucune image trouvée dans le dossier '{selected_class}'.")
            else:
                num_columns = 3
                cols = st.columns(num_columns)
                display_size = (200, 200)

                for index, image_file in enumerate(st.session_state.selected_images):
                    col = cols[index % num_columns]
                    image_path = os.path.join(test_folder, image_file)
                    with col:
                        img = Image.open(image_path)
                        img = img.resize(display_size)
                        st.image(img)

                        if st.button("Prédire", key=f"predict_{image_file}"):
                            predicted_label, predicted_prob = model_prediction(image_path)
                            st.success(f"Prédiction: {predicted_label}")
                            st.progress(predicted_prob / 100)
                            st.write(f"Confiance: {predicted_prob:.2f}%")
