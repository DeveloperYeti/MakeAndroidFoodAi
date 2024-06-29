import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# CSV 파일 경로
file_path = './MENU_DSCRN_INFO_KOREAN.csv'

# 데이터 유형 지정
dtype_spec = {
    'Menu_num': str,
    'Menu_name': str,
    'Menu_ex': str,
    'Menu': str,
    'Menu_detail': str
}

# CSV 파일 읽기
data = pd.read_csv(file_path, dtype=dtype_spec)

# 필요한 열 선택 및 결측값 처리
data = data[['Menu_num', 'Menu_name', 'Menu_ex', 'Menu', 'Menu_detail']]
data = data.dropna()

# 두 개의 텍스트 열을 결합
data['combined_text'] = data['Menu_ex'] + ' ' + data['Menu_detail']

# 모델 로드
loaded_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
loaded_model.load_state_dict(torch.load('Model'))
loaded_model.eval()

while True:
    # 사용자 입력 받기
    user_input = input("어떤 느낌의 음식을 먹고싶어? (종료하려면 'exit' 입력) ")

    if user_input.lower() == 'exit':
        print("프로그램을 종료합니다.")
        break

    # 사용자 입력에 대한 문장 임베딩 생성
    user_embedding = loaded_model.encode(user_input, convert_to_tensor=True)

    # 음식 설명에 대한 문장 임베딩 생성
    embeddings = loaded_model.encode(data['combined_text'].tolist(), convert_to_tensor=True)

    # 유사도 계산
    cosine_scores = util.pytorch_cos_sim(user_embedding, embeddings)[0]

    # 가장 유사한 음식 추천
    top_k = min(1, len(data))  # 추천할 상위 k개
    top_results = torch.topk(cosine_scores, k=top_k)

    # 추천 결과 출력
    print("추천 결과:")
    for score, idx in zip(top_results[0], top_results[1]):
        idx = idx.item()  # LongTensor를 정수로 변환
        print(f"Menu Name: {data.iloc[idx]['Menu_name']}, Menu Description: {data.iloc[idx]['combined_text']}, Score: {score.item()}")

    # 추가적인 음식 추천 여부 확인
    response = input("더 먹고싶은거 있어? (있으면 'yes', 없으면 'exit' 입력) ")
    if response.lower() == 'exit':
        print("프로그램을 종료합니다.")
        break
    elif response.lower() != 'yes':
        print("잘못된 입력입니다. 프로그램을 종료합니다.")
        break
