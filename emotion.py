from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
import joblib
# Tải bộ dữ liệu GoEmotions
dataset = load_dataset("go_emotions")

# Lấy danh sách nhãn cảm xúc từ GoEmotions
emotion_labels = dataset['train'].features['labels'].feature.names

# Tạo tokenizer và mô hình DistilBERT
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Tiền xử lý văn bản và trích xuất đặc trưng từ DistilBERT
def extract_features(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Sử dụng embedding của token [CLS] (thường là outputs.last_hidden_state[:, 0])
    return outputs.last_hidden_state[:, 0].numpy()

# Chuyển đổi tất cả các văn bản trong tập huấn luyện thành đặc trưng
texts = [item['text'] for item in dataset['train']]
features = extract_features(texts)

# Chuyển nhãn số thành tên nhãn
labels = [emotion_labels[label[0]] for label in dataset['train']['labels']]

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Áp dụng SMOTE để cân bằng dữ liệu
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Tạo và huấn luyện mô hình Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_resampled, y_train_resampled)

# Dự đoán và đánh giá mô hình
y_pred = nb_model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Lưu mô hình Naive Bayes và tokenizer
joblib.dump(nb_model, 'naive_bayes_model.pkl')
joblib.dump(tokenizer, 'tokenizer.pkl')

print("Mô hình Naive Bayes và tokenizer đã được lưu thành công.")
