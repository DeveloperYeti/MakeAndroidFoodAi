import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
dtype_spec = {
    'Menu_num': str,
    'Menu_name': str,
    'Menu_ex': str,
    'Menu': str,
    'Menu_detail': str
}

def recommend_food(description, model, vectorizer, data, n_recommendations=5):
    # 입력한 설명을 벡터화
    description_vector = vectorizer.transform([description])

    # 가장 가까운 이웃 찾기
    distances, indices = model.kneighbors(description_vector, n_neighbors=n_recommendations)

    # 추천 결과 반환
    recommended_indices = indices[0]
    recommended_foods = data.iloc[recommended_indices]

    return recommended_foods

# CSV 파일 경로
file_path = './MENU_DSCRN_INFO_KOREAN_with_keywords.csv'

data = pd.read_csv(file_path, dtype=dtype_spec)

# CSV 파일 읽기
data = pd.read_csv(file_path, dtype=dtype_spec)

# 필요한 열 선택 및 결측값 처리
data = data[['Menu_num', 'Menu_name', 'Menu_ex', 'Menu', 'Menu_detail']]
data = data.dropna()

# 두 개의 텍스트 열을 결합하여 벡터화
data['combined_text'] = data['Menu_ex'] + ' ' + data['Menu_detail']

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['combined_text'])

# 벡터화된 데이터 확인
print(X.shape)

# KNN 모델 학습
knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(X)

# 예시: 사용자가 입력한 음식 설명을 바탕으로 추천
user_input = input("어떤 느낌의 음식을 먹고싶어?")

# 추천 결과를 제공하는 함수
def recommend_food(description, model, vectorizer, data, n_recommendations=5):
    # 입력한 설명을 벡터화
    description_vector = vectorizer.transform([description])

    # 가장 가까운 이웃 찾기
    distances, indices = model.kneighbors(description_vector, n_neighbors=n_recommendations)

    # 추천 결과 반환
    recommended_indices = indices[0]
    recommended_foods = data.iloc[recommended_indices]

    return recommended_foods

# 사용자가 입력한 값을 기준으로 추천 결과 출력
recommendations = recommend_food(user_input, knn, vectorizer, data)

print(recommendations)