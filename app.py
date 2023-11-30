import tempfile
import os
import streamlit as st
from cv import *

st.title("My Cv(Curriculum Vitae) tutor")
st.markdown("<a href='https://github.com/jh941213' style='color: gray;'>Made by 미남홀란드</a>", unsafe_allow_html=True)
st.markdown("PDF 파일을 업로드하고 관련 질문을 입력하세요.")

with st.sidebar:
    st.header("사용 방법")
    st.text("1. PDF 파일을 업로드합니다.\n2. 본인 이력에 관련하여 질문을 합니다..\n3. 결과를 확인합니다.")


uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")


user_query = st.text_input("본인 이력서를 토대로 질문을 입력하세요")
submit_button = st.button("전송")

if submit_button:
    # 사용자가 전송 버튼을 클릭하면 이 부분이 실행됩니다.
    # user_query 변수를 사용하여 필요한 작업을 수행하세요.
    st.write("입력된 질문:", user_query)
    # 추가 작업 수행...


if uploaded_file is not None:
    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        # 업로드된 파일의 내용을 임시 파일에 쓴다
        print("----------------------")
        print("파일전체이름"+ tmp_file.name)
        tmp_file.write(uploaded_file.getvalue())
        
        file_path = tmp_file.name
        fpath = os.path.dirname(file_path) 
        fpath = fpath + "/" # 상위 디렉토리 경로를 얻습니다.
        fname = os.path.basename(file_path)  # 파일 이름을 얻습니다.

        print("최종확인 : fpath", fpath)
        print("최종확인 : fname", fname)
       

    print("*****************h")
    loader = PyPDFLoader(file_path)
    print(loader)
    
    docs = loader.load()
    tables = [] 
    texts = [d.page_content for d in docs]

    raw_pdf_elements = extract_pdf_elements(fpath, fname)

    texts, tables = categorize_elements(raw_pdf_elements)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000, chunk_overlap=0
    )
    joined_texts = " ".join(texts)
    texts_4k_token = text_splitter.split_text(joined_texts)

    text_summaries, table_summaries = generate_text_summaries(
        texts_4k_token, tables, summarize_texts=True
    )
    img_base64_list, image_summaries = generate_img_summaries(fpath)

    vectorstore = Chroma(
        collection_name="mm_rag_jh_blog", embedding_function=OpenAIEmbeddings(openai_api_key="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")
    )

    # Create retriever
    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        texts,
        table_summaries,
        tables,
        image_summaries,
        img_base64_list,
    )

    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

    response = chain_multimodal_rag.invoke(user_query)

    st.write(response)