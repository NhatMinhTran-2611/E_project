import joblib
import numpy as np
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import json
import os

# Load các file JSON
def load_json_files():
    with open('data/ekman_mapping.json', 'r', encoding='utf-8') as f:
        ekman_mapping = json.load(f)
    with open('data/emotion_keywords.json', 'r', encoding='utf-8') as f:
        emotion_keywords = json.load(f)
    with open('data/extended_emotion_labels.json', 'r', encoding='utf-8') as f:
        extended_emotion_labels = json.load(f)
    with open('data/sentiment_mapping.json', 'r', encoding='utf-8') as f:
        sentiment_mapping = json.load(f)
    return ekman_mapping, emotion_keywords, extended_emotion_labels, sentiment_mapping

ekman_mapping, emotion_keywords, extended_emotion_labels, sentiment_mapping = load_json_files()

# Load mô hình và tokenizer
nb_model = joblib.load('naive_bayes_model.pkl')
tokenizer = DistilBertTokenizer.from_pretrained('./tokenizer')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Hàm trích xuất đặc trưng từ văn bản bằng DistilBERT
def extract_features(texts, batch_size=8):
    features = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        features.append(outputs.last_hidden_state[:, 0].numpy())
    return np.vstack(features)

# D
