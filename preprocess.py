
import os
import pandas as pd
import regex
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from string import punctuation


def load_dataset(data_path="./data", categories=None):
    """
    Load text documents from specified categories into a pandas DataFrame.
    
    Args:
        data_path (str): Path to the data directory
        categories (list): List of category folders to load. If None, loads ["Crime", "Entertainment", "Politics", "Science"]
    
    Returns:
        pd.DataFrame: DataFrame with columns ['ID', 'Category', 'Content']
    """
    if categories is None:
        categories = ["Crime", "Entertainment", "Politics", "Science"]
    
    data = []
    
    for folder in categories:
        files = os.listdir(os.path.join(data_path, folder))
        for file in files:
            try:
                with open(os.path.join(data_path, folder, file)) as f:
                    contents = " ".join(f.readlines())
                    data.append([file.split(".")[0], folder, contents])
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                continue
    
    return pd.DataFrame(data, columns=['ID', 'Category', 'Content'])

    
# Limpieza de Texto
def clean_text(text, stop_words):
    """
        Preprocesamiento:
            limpieza, 
            tokenización, 
            lematización
    """

    wordnet_lemmatizer = WordNetLemmatizer()

    try:
        tokens = WordPunctTokenizer().tokenize(text.lower())
        filtered = [
            regex.sub(r'\p{^Latin}', '', w)  # Filtrar caracteres
            for w in tokens if w.isalpha() and len(w) > 3
        ]
        
        filtered = [
            wordnet_lemmatizer.lemmatize(w, pos="v") 
            for w in filtered if w not in stop_words
        ]
        return " ".join(filtered)

    except Exception as e:
        print(f"Error procesando texto: {text[:50]}... {e}")
        return ""

def clean_dataset(df):
    stop = set(stopwords.words("english") + list(punctuation))
    custom_stopwords = {"email", "subject", "re", "fw", "https", "www"}
    stop.update(custom_stopwords)
    # Limpieza al dataset
    df["FixedText"] = df["Content"].apply(lambda x: clean_text(x, stop))
    return df


# Vectorización
# Convertir texto en representaciones numéricas (TF-IDF)

def vectorize_dataset(df):
    tfidf = TfidfVectorizer(max_features=5000, lowercase=False)
    X = tfidf.fit_transform(df["FixedText"])
    y = df["Category"]
    return X, y, tfidf

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from collections import Counter

def balance_dataset(X, y, tfidf, target_counts=None):
    """
    Balance dataset using SMOTE and RandomUnderSampler.
    
    Args:
        X: Feature matrix (sparse or dense)
        y: Target labels
        target_counts (dict): Target count for each category. If None, uses default strategy
    
    Returns:
        tuple: (balanced_df, y_resampled) - Balanced feature DataFrame and labels
    """
    if target_counts is None:
        target_counts = {
            "Science": 2500,       # Reducir
            "Politics": 2500,      # Reducir
            "Crime": 2000,         # Aumentar
            "Entertainment": 2000  # Aumentar
        }
    
    # SMOTE for underrepresented classes
    smote = SMOTE(
        sampling_strategy={k: v for k, v in target_counts.items() if v > Counter(y)[k]},
        random_state=42
    )
    X_smote, y_smote = smote.fit_resample(X, y)
    
    # Undersample overrepresented classes
    undersample = RandomUnderSampler(
        sampling_strategy={k: v for k, v in target_counts.items() if v < Counter(y_smote)[k]},
        random_state=42
    )
    X_resampled, y_resampled = undersample.fit_resample(X_smote, y_smote)
    
    # Create balanced DataFrame
    balanced_df = pd.DataFrame.sparse.from_spmatrix(
        X_resampled, 
        columns=tfidf.get_feature_names_out()
    )
    
    return balanced_df, y_resampled

def get_preprocessed_dataset(data_path="./data", categories=None):
    df = load_dataset(data_path, categories)
    df = clean_dataset(df)
    X, y, tfidf = vectorize_dataset(df)
    balanced_df, y_resampled = balance_dataset(X, y, tfidf)
    return balanced_df, y_resampled


