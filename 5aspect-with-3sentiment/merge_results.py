import os
import glob

print("여러 결과 파일을 하나로 병합하는 중...")

# 결과 파일들 찾기
pattern = "dataset/*_5aspect_3sentiment*.jsonl"
input_files = glob.glob(pattern)

if not input_files:
    print(f"병합할 파일을 찾을 수 없습니다. 패턴: {pattern}")
    exit(1)

print(f"병합할 파일 {len(input_files)}개 발견:")
for file in input_files:
    file_size = os.path.getsize(file) / 1024 / 1024
    print(f"- {file} ({file_size:.2f} MB)")

# 출력 파일 경로
output_file = "dataset/review_5up_5aspect_3sentiment_merged.jsonl"

# 파일 병합
total_lines = 0
with open(output_file, "w", encoding="utf-8") as outfile:
    for filename in input_files:
        print(f"파일 병합 중: {filename}")
        file_lines = 0
        with open(filename, "r", encoding="utf-8") as infile:
            for line in infile:
                outfile.write(line)
                file_lines += 1
                total_lines += 1
        print(f"- {file_lines}개 라인 병합됨")

# 결과 확인
output_size = os.path.getsize(output_file) / 1024 / 1024
print(f"병합 완료! 총 {total_lines}개 리뷰 결과가 병합되었습니다.")
print(f"결과 파일: {output_file} ({output_size:.2f} MB)")
