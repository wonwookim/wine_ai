### 시스템 아키텍쳐
    - Frontend
        - Streamlit 기반
        - 이미지 업로드, 설명 입력, 결과 표시(리뷰, 유사도, 상세 추천)
    - Backend
        - sommelier.py
            - 이미지 분석, 벡터 DB 검색, LLM 프롬프트 처리
    - 외부 서비스
        - OpenAI API: GPT-4o mini(LLM), exte-embedding-3-small(임베딩)
    - 환경 설정
        - .env 파일로 처리
### 주요 사용 기술
    - 주요 라이브러리
        - openai, langchain_openai, ...
    - 모델
        - 토큰 임베딩: exte-embedding-3-small
        - LLM: GPT-4o mini
    - 벡터 DB
        - Pinecone(us-east1-aws, cosine metric, dimension = 1536)