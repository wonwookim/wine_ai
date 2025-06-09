import streamlit as st
from PIL import Image
import io, base64
from sommelier import search_wine, recommand_wine, describe_dish_flavor
st.title("Somelier AI")

col1, col2 = st.columns([3, 1])
with col1:
    uploaded_image = st.file_uploader('요리 이미지를 업로드 하세요', type = ['jpg', 'jpeg', 'png'])
    user_prompt = st.text_input('프롬프트를 입력하세요.', '이 요리에 어울리는 와인을 추천해주세요.')

with col2:
    if uploaded_image:
        st.image(uploaded_image, caption = '업로드된 요리 이미지', use_container_width = True)

import time
with col1:
    if st.button('추천하기'):
        if not uploaded_image:
            st.warning('이미지를 업로드 해주세요.')
        else:
            with st.spinner('1단계: 요리의 맛과 향을 분석하는 중'):
                # 멀티모달 모델을 이용하여 사진의 요리의 특성을 분석
                # 출력
                time.sleep(3)
                st.markdown('### 요리의 맛과 향 분석 결과')
                 # 1) 업로드된 파일 읽기
                # UploadedFile 객체에서 이미지 로드
                img = Image.open(uploaded_image)

                # RGBA → RGB 변환 후 해상도 축소
                img = img.convert("RGB").resize((512, 512))

                # JPEG로 버퍼에 저장
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                b64 = base64.b64encode(buffer.getvalue()).decode()

                dish_flavor = describe_dish_flavor(b64, '해당 이미지의 요리의 맛과 향을 분석해줘')
                st.info(dish_flavor)
            with st.spinner('2단계: 요리에 어울리는 와인 리뷰를 검색하는 중'):
                # 요리의 특성 정보로 와인을 추천하는 ai 동작
                # 출력
                time.sleep(3)
                st.markdown('### 와인 리뷰 검색 결과')
                wine_search_result = search_wine(dish_flavor)
                st.text(wine_search_result['wine_reviews'])
            with st.spinner('3단계: AI 소믈리에가 와인 페어링에 대한 추천 글을 생성하는 중'
                            ):
                # LLM을 이용해서 추천 글 생성
                time.sleep(3)
                st.markdown('### AI 소믈리에 와인 페어링 추천')
                wine_recommandation = recommand_wine(inputs= wine_search_result)
                st.info(wine_recommandation)
            st.success('추천이 완료되었습니다.')
# 유사도를 높이도록 --
# 추천된 와인의 이미지를 생성 또는 찾아서 출력해보기