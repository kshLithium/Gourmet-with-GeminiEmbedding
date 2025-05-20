# gourmet
# 임시 (수정할 예정)

# 🍽️ 설명 가능한 맛집 추천 시스템

사용자 평점과 리뷰 텍스트의 일관성을 고려한, 설명 가능한 맛집 추천 시스템입니다.  
기본 협업 필터링(Item-based CF)과 감성 정보 기반 하이브리드 CF(Hybrid CF)를 지원합니다.

---

## 📁 주요 파일 설명

| 파일명 | 설명 |
|--------|------|
| `run_experiment.py` | 전체 추천 파이프라인 실행 (데이터 로드 → 유사도 계산 → 추천 → 평가) |
| `data_loader.py` | 평점 데이터 또는 감성 벡터 포함 데이터를 로드하는 함수 |
| `cf_utils.py` | 협업 필터링을 위한 유틸 함수 (Leave-One-Out, 유사도 계산 등) |
| `eval_metrics.py` | Precision, Recall, NDCG 등의 추천 성능 지표 구현 |
| `similarity.py` | 코사인 유사도 계산 함수 (dict 기반 / vector 기반) |
| `hybrid_cf.py` | 감성 벡터 기반 하이브리드 CF 구성 함수 |

---

## 📁 디렉토리 구조
```
├── notebooks/ # 실험 코드
│ └──  experiment.ipynb - 실험

├── src/ # 프로젝트 소스 코드
│ ├── experiment_runner.py - 전체 추천 파이프라인 실행 스크립트
│ ├── data_loader.py - 평점 / 감성 데이터 로드 함수
│ ├── cf_utils.py - CF 유틸 함수 (유사도 계산, 추천 등)
│ ├── hybrid_cf.py - 감성 기반 하이브리드 추천 로직
│ ├── similarity.py - 코사인 유사도 함수
│ └── eval_metrics.py - 평가 지표 함수

├── data/ # 데이터 디렉토리
│ ├── raw/ - 원본 JSON 파일
│ └── review_5up_5aspect_3sentiment_vectorized_clean.json

├── README.md # 프로젝트 설명 파일
├── step1.ipynb - 원본 데이터 로드 및 클렌징
├── step2.ipynb - merged_dataset 생성
└── train_test.ipynb - 추천용 데이터셋 구성 및 저장
```