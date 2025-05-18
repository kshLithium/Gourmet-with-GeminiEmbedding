import os

# 파일을 2:1:1로 나누는 코드
print("파일을 2:1:1 비율로 나누는 중...")

# 입력 파일 경로
input_file = "dataset/review_5up.json"

# 파일 크기 확인
file_size = os.path.getsize(input_file)
print(f"입력 파일 크기: {file_size/1024/1024:.2f} MB")

# 전체 라인 수 확인
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

total_lines = len(lines)
print(f"총 리뷰 수: {total_lines}")

# 2:1:1 비율로 분할 계산
part1_ratio = 0.5  # 2/4 = 0.5
part2_ratio = 0.25  # 1/4 = 0.25
part3_ratio = 0.25  # 1/4 = 0.25

part1_lines = int(total_lines * part1_ratio)
part2_lines = int(total_lines * part2_ratio)
part3_lines = total_lines - part1_lines - part2_lines  # 나머지는 마지막 파일에

# 각 파트의 시작과 끝 인덱스
part1_start, part1_end = 0, part1_lines
part2_start, part2_end = part1_end, part1_end + part2_lines
part3_start, part3_end = part2_end, total_lines

# 파일명 생성
output_file1 = f"dataset/review_5up_part1_50pct_{part1_start+1}-{part1_end}.json"
output_file2 = f"dataset/review_5up_part2_25pct_{part2_start+1}-{part2_end}.json"
output_file3 = f"dataset/review_5up_part3_25pct_{part3_start+1}-{part3_end}.json"

# 파일 나누기
print(f"파트 1 (50%): {part1_start+1}~{part1_end} ({part1_lines}개 리뷰)")
with open(output_file1, "w", encoding="utf-8") as f:
    f.writelines(lines[part1_start:part1_end])

print(f"파트 2 (25%): {part2_start+1}~{part2_end} ({part2_lines}개 리뷰)")
with open(output_file2, "w", encoding="utf-8") as f:
    f.writelines(lines[part2_start:part2_end])

print(f"파트 3 (25%): {part3_start+1}~{part3_end} ({part3_lines}개 리뷰)")
with open(output_file3, "w", encoding="utf-8") as f:
    f.writelines(lines[part3_start:part3_end])

print("파일 분할 완료!")
print(f"파일 1: {output_file1}")
print(f"파일 2: {output_file2}")
print(f"파일 3: {output_file3}")
