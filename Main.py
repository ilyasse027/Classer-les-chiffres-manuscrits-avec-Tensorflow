import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string

# les ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Chargement et prétraitement des données MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255


train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)


# Fonction pour prétraiter le texte
def preprocess_text(text):
    # Tokenisation
    tokens = word_tokenize(text.lower())
    # Suppression des stopwords et de la ponctuation
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


# Génération de textes fictifs pour chaque chiffre
digit_texts = [
    "zero circular shape",
    "one single vertical line",
    "two curved line with horizontal base",
    "three curved lines",
    "four vertical and horizontal lines",
    "five top curve and bottom hook",
    "six circular top with vertical line",
    "seven angled line",
    "eight double circular shape",
    "nine circular top with curved tail"
]

# Prétraitement des textes
processed_texts = [preprocess_text(text) for text in digit_texts]

# Création d'un modèle Word2Vec
word2vec_model = Word2Vec(sentences=processed_texts, vector_size=50, window=5, min_count=1, workers=4)


# Fonction pour obtenir le vecteur Word2Vec
def get_avg_word2vec(tokens, model, vector_size=50):
    vec = np.zeros(vector_size)
    count = 0
    for word in tokens:
        try:
            vec += model.wv[word]
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# Création des vecteurs Word2Vec
digit_vectors = np.array([get_avg_word2vec(tokens, word2vec_model) for tokens in processed_texts])


def create_model(input_shape, text_vector_size=50):
    # Branche pour l'image
    image_input = layers.Input(shape=input_shape)
    x = layers.Flatten()(image_input)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    # Branche pour le texte
    text_input = layers.Input(shape=(text_vector_size,))
    y = layers.Dense(32, activation='relu')(text_input)

    # Concaténation des branches
    combined = layers.concatenate([x, y])

    # Couches finales
    z = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(10, activation='softmax')(z)

    model = models.Model(inputs=[image_input, text_input], outputs=output)
    return model



# Création et compilation du modèle
model = create_model((28, 28))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Création de vecteurs textuels factices pour l'entraînement
dummy_text_vectors = np.random.rand(len(train_images), 50)

# Entraînement du modèle
history = model.fit([train_images, dummy_text_vectors], train_labels, epochs=10,
                    validation_split=0.1, batch_size=32, verbose=1)

# Évaluation du modèle
dummy_test_vectors = np.random.rand(len(test_images), 50)
test_loss, test_acc = model.evaluate([test_images, dummy_test_vectors], test_labels, verbose=0)
print(f'Précision sur l\'ensemble de test : {test_acc:.4f}')

# Prédictions
predictions = model.predict([test_images, dummy_test_vectors])
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Matrice de confusion
cm = confusion_matrix(true_labels, predicted_labels)

# Visualisation de la matrice de confusion
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion')
plt.ylabel('Vraie étiquette')
plt.xlabel('Étiquette prédite')
plt.show()

# Rapport de classification
print(classification_report(true_labels, predicted_labels))

# Visualisation de l'historique
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Précision (entraînement)')
plt.plot(history.history['val_accuracy'], label='Précision (validation)')
plt.title('Précision du modèle')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte (entraînement)')
plt.plot(history.history['val_loss'], label='Perte (validation)')
plt.title('Perte du modèle')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()

plt.tight_layout()
plt.show()


# Fonction pour prédire un chiffre
def predict_with_nlp(image, text):
    image = np.expand_dims(image, axis=0)
    text_vec = get_avg_word2vec(preprocess_text(text), word2vec_model)
    text_vec = np.expand_dims(text_vec, axis=0)

    combined_pred = model.predict([image, text_vec])[0]
    return np.argmax(combined_pred)


# Exemple
test_index = 42
test_image = test_images[test_index]
test_text = "curved line with horizontal base"  # Description textuelle du chiffre 2
predicted_digit = predict_with_nlp(test_image, test_text)

plt.imshow(test_image.reshape(28, 28), cmap='gray')
plt.title(f'Prédiction: {predicted_digit}')
plt.axis('off')
plt.show()

print(f"Chiffre prédit: {predicted_digit}")
print(f"Vrai chiffre: {np.argmax(test_labels[test_index])}")