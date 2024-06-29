import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import tensorflow as tf

# CSV 파일 로드
file_path = './TB_RECIPE_SEARCH-20231130.csv'
data = pd.read_csv(file_path)

# 필요한 열 선택 및 결측값 처리
data_subset = data[['RCP_TTL', 'CKG_NM', 'CKG_MTH_ACTO_NM']].fillna('')

# 랜덤으로 1000개 샘플링
sampled_data = data_subset.sample(n=1000, random_state=42).reset_index(drop=True)

# 각 레시피의 관련 열을 하나의 문자열로 결합
sampled_data['combined'] = sampled_data[['RCP_TTL', 'CKG_NM', 'CKG_MTH_ACTO_NM']].agg(' '.join, axis=1)

# TF-IDF 벡터화기 초기화
vectorizer = TfidfVectorizer()

# 결합된 텍스트 데이터로 TF-IDF 행렬 생성
tfidf_matrix = vectorizer.fit_transform(sampled_data['combined'])

# 레시피 추천 함수
def recommend_recipes(user_input, top_n=3):
    user_tfidf = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    recommendations = sampled_data.iloc[top_indices][['RCP_TTL', 'CKG_NM', 'CKG_MTH_ACTO_NM']]
    return recommendations

# 사용 예시
user_input = input("무슨 요리를 원해요? ")
recommended = recommend_recipes(user_input)
print(recommended)

# 모델과 벡터화기 저장
model_file_path = './recipe_recommender.pkl'
with open(model_file_path, 'wb') as file:
    pickle.dump((vectorizer, sampled_data), file)

print(f"Model saved to {model_file_path}")

# TensorFlow Lite 모델 변환 (더미 모델 사용)
class DummyModel(tf.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def predict(self, input_text):
        return tf.constant([[0.1, 0.9]])

dummy_model = DummyModel()

# TFLite 모델로 변환
converter = tf.lite.TFLiteConverter.from_concrete_functions([dummy_model.predict.get_concrete_function()])
tflite_model = converter.convert()

# TFLite 모델 저장
tflite_model_path = './recipe_recommender.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved to {tflite_model_path}")
