import pandas as pd
import nltk
import numpy as np
import spacy
from collections import defaultdict
from pandas import DataFrame
from nltk.corpus import stopwords, wordnet as wn
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support


#Definiamo gli oggetti da riutilizzare successivamente
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
nlp = spacy.load("C:\\Users\\valer\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\en_core_web_sm\\en_core_web_sm-3.7.1")
#Ho dovuto specificare il percorso assoluto per caricare il modello di Spacy. Scrivendo solo il nome non riesco a caricarlo probabilmente per qualche errore durante l'installazione.

#<-----------------------------STEP 1: DATA PRE PROCESSING <CON> LEMMATIZZAZIONE E RIMOZIONE STOPWORDS ------------------------------------>
def pre_processing_avanzato(dataset, split):
    # Creazione del DataFrame utilizzando le colonne 'text' e 'label'
    data_avanzato = pd.DataFrame({'text': dataset[split]['text'], 'label': dataset[split]['label']})
    # Creazione della colonna 'label_text' con etichette testuali
    data_avanzato['label_text'] = [my_labels[l] for l in data_avanzato['label']]
    # Eliminazione dei duplicati
    data_avanzato = data_avanzato.drop_duplicates()
    tweet_processati = []
    for text in data_avanzato['text']:
        words = nltk.word_tokenize(text.lower())
        # Rimozione delle stopwords e lemmatizzazione con list comprehension
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        tweet_processato = ' '.join(words)
        tweet_processati.append(tweet_processato)
    data_avanzato['text'] = tweet_processati

    # Visualizzazione delle statistiche
    print(f"Numero di elementi nel DataFrame: {len(data_avanzato)}")
    print({k: len(data_avanzato[data_avanzato['label'] == k]) for k in set(data_avanzato['label'])})
    return data_avanzato

dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
my_labels = {0: "Negative", 1: "Positive", 2: "Neutral"}


# Pre-elaborazione del set di validazione (usato come set di test) dato che manca un vero e proprio set di test nel nostro dataset;
test_avanzato = pre_processing_avanzato(dataset, 'validation')
train_avanzato = pre_processing_avanzato(dataset, 'train')
print(test_avanzato)
print("\n")
print(train_avanzato)

#<---------------------------- DATA PRE PROCESSING <SENZA> LEMMATIZZAZIONE E RIMOZIONE STOPWORDS ------------------------------------>
def pre_processing_ridotto(dataset, split):
    data_ridotto = pd.DataFrame({'text': dataset[split]['text'], 'label': dataset[split]['label']})
    data_ridotto['label_text'] = [my_labels[l] for l in data_ridotto['label']]
    data_ridotto = data_ridotto.drop_duplicates()
    return data_ridotto

test_ridotto = pre_processing_ridotto(dataset, 'validation')
train_ridotto = pre_processing_ridotto(dataset, 'train')


#<-----------------------------STEP 2: GENERAZIONE EMBEDDINGS ------------------------------------>
#Generiamo gli embeddings
train_embeddings_avanzato = model.encode(train_avanzato['text'])
test_embeddings_avanzato = model.encode(test_avanzato['text'])

train_embeddings_ridotto = model.encode(train_ridotto['text'])
test_embeddings_ridotto = model.encode(test_ridotto['text'])


#<-----------------------------STEP 3: TRAIN MODELLO ------------------------------------>
#Alleniamo il modello con la funzione fit
mlp = MLPClassifier(hidden_layer_sizes=(1000,))
mlp.fit(train_embeddings_avanzato, train_avanzato["label"])


#<-----------------------------STEP 4: VALIDAZIONE ------------------------------------>
test_predictions_avanzato = mlp.predict(test_embeddings_avanzato)
print("Le statitistiche del classificatore EFFETTUANDO lemmatizzazione e rimozione stopwords sono: ")
report_avanzato = classification_report(test_avanzato['label'], test_predictions_avanzato, target_names=my_labels.values(), output_dict=False)
#Nel classification report ho lasciato i valori di default visualizzabili nella documentazione. Ho solo modificato i nomi delle variabili con le mie etichette my_labels (Quindi Positive, Negative e Neutral)
print(report_avanzato)

test_predictions_ridotto = mlp.predict(test_embeddings_ridotto)
print("Le statitistiche del classificatore NON EFFETTUANDO lemmatizzazione e rimozione stopwords sono: ")
report_ridotto = classification_report(test_ridotto['label'], test_predictions_ridotto, target_names=my_labels.values(), output_dict=False)
print(report_ridotto)


#<-----------------------------STEP 5: NAMED ENTITY RECOGNITION ------------------------------------>

def named_entity_recognition(data):
    entity_counts = defaultdict(int)
    negative_entity_counts = defaultdict(int)
    #Utilizzo due default dict per economia di codice

    for i in range(len(data)):
        text = data['text'][i]
        label_text = data['label_text'][i]
        doc = nlp(text)
        for entity in doc.ents:
            entity_counts[entity.label_] += 1
            if label_text == 'Negative':
                negative_entity_counts[entity.label_] += 1

    entity_percentages = {}
    for label in entity_counts:
        if entity_counts[label] > 0:
            entity_percentages[label] = int((negative_entity_counts[label] / entity_counts[label]) * 100)
        else:
            entity_percentages[label] = 0

    return entity_percentages


negative_entity_percentages = named_entity_recognition(test_avanzato)

print("Percentuali delle entità nominate etichettate come 'Negative'")
print(negative_entity_percentages)


#<-----------------------------STEP 6: TESTING UTENTE ------------------------------------>
print("Inserisci la frase o digita 'stop' per interrompere: ")
while True:
    frase = input()
    if frase.lower() == "stop":                                                             #Uso la funzione lower per evitare lettere maiuscole ed eventuali problemi
        break
    prediction = mlp.predict(model.encode([frase]))[0]                                      #Inseriamo 0 perchè lui restituisce una lista, in questo caso abbiamo solo un elemento
    print(f"Il sentimento della frase '{frase}' è:", my_labels[prediction])