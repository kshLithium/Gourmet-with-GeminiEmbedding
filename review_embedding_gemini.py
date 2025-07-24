import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd
import json
import time
from tqdm import tqdm

# google.api_core.exceptions에서 특정 에러를 import 합니다.
from google.api_core import exceptions

# --- 1. 환경 설정 및 API 키 구성 ---
load_dotenv()
api_key = os.getenv("GEMENI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMENI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요."
    )

genai.configure(api_key=api_key)
model_name = "gemini-embedding-001"

# --- 2. 파일 및 배치 설정 ---
input_file = "./Dataset/review_business_5up_cleaned.jsonl"
output_file = "./Dataset/review_business_5up_with_embedded_vector.jsonl"
BATCH_SIZE = 100
DELAY_BETWEEN_REQUESTS = 10  # 정상적인 요청 사이의 대기 시간 (초))
RETRY_DELAY = 30  # 429 에러 발생 시 대기 시간 (초)
MAX_RETRIES = 50  # 최대 재시도 횟수


# --- 3. 메인 실행 함수 (재시도 및 이어하기 기능 추가) ---
def process_and_embed_reviews_resumable():
    print(f"입력 파일에서 데이터를 로드합니다: {input_file}")
    try:
        df_reviews = pd.read_json(input_file, lines=True)
        review_texts = df_reviews["text"].astype(str).tolist()
        total_reviews = len(review_texts)
        print(f"총 {total_reviews}개의 리뷰를 로드했습니다.")
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{input_file}'을 찾을 수 없습니다.")
        return

    # --- 이어하기 로직 ---
    start_index = 0
    if os.path.exists(output_file):
        try:
            # 기존 출력 파일의 줄 수를 세어 어디까지 처리되었는지 확인
            with open(output_file, "r", encoding="utf-8") as f:
                processed_count = sum(1 for line in f)
            if processed_count > 0:
                start_index = processed_count
                print(
                    f"기존 파일 '{output_file}'에 {processed_count}개의 임베딩이 저장되어 있습니다."
                )
                print(f"{processed_count}번째 리뷰부터 이어합니다.")
        except Exception as e:
            print(f"기존 출력 파일 처리 중 오류 발생: {e}. 처음부터 다시 시작합니다.")
            start_index = 0

    # 파일을 추가 모드('a') 또는 쓰기 모드('w')로 엽니다.
    file_mode = "a" if start_index > 0 else "w"
    with open(output_file, file_mode, encoding="utf-8") as f_out:

        # tqdm의 시작 지점을 start_index로 설정
        for i in tqdm(
            range(start_index, total_reviews, BATCH_SIZE),
            initial=start_index,
            total=total_reviews,
            desc="임베딩 진행률",
        ):

            batch_texts = review_texts[i : i + BATCH_SIZE]
            retries = 0

            while retries < MAX_RETRIES:
                try:
                    result = genai.embed_content(
                        model=model_name,
                        content=batch_texts,
                        task_type="SEMANTIC_SIMILARITY",
                        output_dimensionality=3072,
                    )

                    embeddings = result["embedding"]

                    batch_df = df_reviews.iloc[i : i + BATCH_SIZE].copy()
                    # 임베딩 리스트의 길이가 배치와 다를 경우를 대비한 안전장치
                    batch_df["embedding"] = embeddings[: len(batch_df)]

                    json_lines = batch_df.to_json(
                        orient="records", lines=True, force_ascii=False
                    )
                    f_out.write(json_lines)
                    f_out.flush()  # 버퍼를 비워 파일에 즉시 쓰도록 함

                    # 성공 시 재시도 루프 탈출
                    break

                except exceptions.ResourceExhausted as e:
                    retries += 1
                    print(
                        f"\n[경고] Quota 초과 (429 에러) (배치 인덱스: {i}). {retries}/{MAX_RETRIES}번째 재시도. {RETRY_DELAY}초 후 다시 시도합니다."
                    )
                    time.sleep(RETRY_DELAY)
                except Exception as e:
                    print(f"\n[오류] 처리 불가 (배치 인덱스: {i}): {e}")
                    # 다른 종류의 에러는 재시도하지 않고 넘어감
                    break

            if retries == MAX_RETRIES:
                print(f"[실패] 배치 인덱스 {i}를 {MAX_RETRIES}번 재시도 후 건너뜁니다.")

            # 다음 배치를 위해 잠시 대기
            time.sleep(DELAY_BETWEEN_REQUESTS)

    print(
        f"\n✅ 작업 완료! 임베딩 결과가 '{output_file}' 파일에 성공적으로 저장되었습니다."
    )


# --- 4. 스크립트 실행 ---
if __name__ == "__main__":
    process_and_embed_reviews_resumable()
