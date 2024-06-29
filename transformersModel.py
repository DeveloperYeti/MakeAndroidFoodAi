import pandas as pd
from sentence_transformers import SentenceTransformer, util
import tensorflow as tf

# 사용 가능한 GPU 확인 및 첫 번째 GPU만 사용하도록 설정
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
else:
    print("No GPU devices found.")

# CSV 파일 경로
file_path = './MENU_DSCRN_INFO_KOREAN.csv'

# CSV 파일 읽기 (일부 데이터만 사용)
dtype_spec = {
    'RCP_TTL': str,
    'CKG_MTRL_CN': str,
    'CKG_KND_ACTO_NM': str,
    'CKG_STA_ACTO_NM': str,
}

# 데이터를 일부만 읽음
data = pd.read_csv(file_path, dtype=dtype_spec, nrows=1000)

# 필요한 열 선택 및 결측값 처리
data = data[['RCP_TTL', 'CKG_MTRL_CN', 'CKG_KND_ACTO_NM', 'CKG_STA_ACTO_NM']].dropna()

# 두 개의 텍스트 열을 결합
data['combined_text'] = data['CKG_MTRL_CN'] + ' ' + data['CKG_KND_ACTO_NM'] + ' ' + data['CKG_STA_ACTO_NM']

# 사전 훈련된 언어 모델 로드
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 예시: 사용자가 입력한 음식 설명을 바탕으로 추천
user_input = input("어떤 음식 먹을랭")  # 디버깅을 위해 고정된 입력 사용


# 사용자 입력에 대한 문장 임베딩 생성
user_embedding = model.encode(user_input, convert_to_tensor=True)

print("사용자 입력 임베딩 생성 완료")

# 모든 레시피에 대한 임베딩 생성
recipe_embeddings = model.encode(data['combined_text'].tolist(), convert_to_tensor=True)

print("레시피 임베딩 생성 완료")

# 유사도 계산
cosine_scores = util.pytorch_cos_sim(user_embedding, recipe_embeddings)[0]

print("유사도 계산 완료")

# 가장 유사한 음식 추천
top_k = min(5, len(data))  # 추천할 상위 k개
top_results = tf.math.top_k(cosine_scores, k=top_k)

print("추천 결과:")
for score, idx in zip(top_results.values.numpy(), top_results.indices.numpy()):
    print(f"Recipe Name: {data.iloc[idx]['RCP_TTL']}, Description: {data.iloc[idx]['combined_text']}, Score: {score}")

# 모델 저장 (Torch 형식)
model.save('sentence_transformer_model')

# TensorFlow 모델 변환
# 사전 훈련된 모델을 TensorFlow 형식으로 변환하기 위한 헬퍼 함수
def convert_to_tf_model(model):
    class ModelWrapper(tf.Module):
        def __init__(self, model):
            self.model = model

        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
        def __call__(self, texts):
            embeddings = self.model.encode(texts.numpy().tolist(), convert_to_tensor=True)
            return embeddings

    return ModelWrapper(model)

# 모델 래핑 및 변환
wrapped_model = convert_to_tf_model(model)

# Concrete function 추출
concrete_function = wrapped_model.__call__.get_concrete_function(tf.TensorSpec(shape=[None], dtype=tf.string))

# SavedModel 형식으로 저장
tf.saved_model.save(wrapped_model, 'saved_model', signatures=concrete_function)
print("TensorFlow 모델이 성공적으로 저장되었습니다.")

# TFLite 모델로 변환
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
tflite_model = converter.convert()

# TFLite 모델 저장
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite 모델이 성공적으로 저장되었습니다.")
