import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd
import json
import time

# tqdm 클래스를 직접 import 합니다.
from tqdm import tqdm

# --- 1. 환경 설정 및 API 키 구성 ---
# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()
api_key = os.getenv("GEMENI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMENI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요."
    )

genai.configure(api_key=api_key)
model_name = "models/embedding-001"

# --- 2. 파일 및 배치 설정 ---
input_file = "./Dataset/review_business_5up_with_text.json"
output_file = "./Dataset/review_business_5up_with_embedded_vector.jsonl"
BATCH_SIZE = 100
DELAY_BETWEEN_REQUESTS = 1  # 1초 대기 (API Quota 방지)


# --- 3. 메인 실행 함수 ---
def process_and_embed_reviews():
    """
    JSONL 파일에서 리뷰를 읽어 Gemini 임베딩을 생성하고,
    원본 데이터에 임베딩을 추가하여 새로운 JSONL 파일로 저장합니다.
    """
    print(f"입력 파일에서 데이터를 로드합니다: {input_file}")
    try:
        df_reviews = pd.read_json(input_file, lines=True)
        review_texts = df_reviews["text"].astype(str).tolist()
        total_reviews = len(review_texts)
        print(f"총 {total_reviews}개의 리뷰를 로드했습니다.")
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{input_file}'을 찾을 수 없습니다.")
        return

    # 출력 파일을 쓰기 모드로 엽니다.
    with open(output_file, "w", encoding="utf-8") as f_out:

        # tqdm을 사용하여 배치 단위로 루프를 실행하고 진행률을 표시합니다.
        # 이제 'tqdm(...)' 호출이 정상적으로 작동합니다.
        for i in tqdm(range(0, total_reviews, BATCH_SIZE), desc="임베딩 진행률"):

            batch_texts = review_texts[i : i + BATCH_SIZE]

            try:
                # API를 호출하여 현재 배치의 텍스트를 임베딩합니다.
                result = genai.embed_content(
                    model=model_name,
                    content=batch_texts,
                    task_type="CLASSIFICATION",
                    output_dimensionality=3072,
                )

                embeddings = result["embedding"]

                # 원본 DataFrame의 해당 배치 부분에 임베딩 결과를 추가합니다.
                batch_df = df_reviews.iloc[i : i + BATCH_SIZE].copy()
                batch_df["embedding"] = embeddings

                # 처리된 배치를 즉시 JSONL 형식으로 파일에 추가합니다.
                json_lines = batch_df.to_json(
                    orient="records", lines=True, force_ascii=False
                )
                f_out.write(json_lines)

            except Exception as e:
                print(f"\n오류 발생 (배치 시작 인덱스: {i}): {e}")
                print("이 배치를 건너뛰고 다음을 계속합니다.")

            finally:
                # API Quota 제한을 피하기 위해 각 요청 사이에 대기합니다.
                time.sleep(DELAY_BETWEEN_REQUESTS)

    print(
        f"\n✅ 작업 완료! 임베딩 결과가 '{output_file}' 파일에 성공적으로 저장되었습니다."
    )


# --- 4. 스크립트 실행 ---
if __name__ == "__main__":
    process_and_embed_reviews()
