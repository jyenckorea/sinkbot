# 1. 베이스 이미지 선택: 가벼운 공식 파이썬 3.11 이미지를 사용합니다.
FROM python:3.11-slim

# 2. 작업 디렉토리 설정: 컨테이너 내부에 /workspace 폴더를 만듭니다.
WORKDIR /workspace

# 3. 라이브러리 설치 (효율적인 캐싱을 위해 requirements.txt만 먼저 복사)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 나머지 소스 코드 전체를 복사합니다.
COPY . .

# 5. Procfile이 실행 명령어를 관리하므로, 여기서는 별도의 CMD가 필요 없습니다.

# ⭐️ 이 서비스들이 어떤 포트를 사용하는지 외부에 알립니다.
EXPOSE 5000
EXPOSE 8501