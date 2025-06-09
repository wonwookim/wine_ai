from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os
load_dotenv()
pc = Pinecone(api_key = os.environ['PINECONE_API_KEY'])
pc.create_index(
    name = os.environ['PINECONE_INDEX_NAME'],
    dimension = 1536,
    metric = 'cosine',
    spec = ServerlessSpec(
        region = 'us-east-1',
        cloud = 'aws'
    )
)
wine_index = pc.Index(os.environ['PINECONE_INDEX_NAME'])
print(wine_index.describe_index_stats())



# csv 파일 읽어서 벡터 db 만들기
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader('winemag-data-130k-v2.csv', encoding = 'utf-8')
docs = loader.load()

# 임베딩 모델
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(
    model = os.environ['OPENAI_EMBEDDING_MODEL'], 
    api_key = os.environ['OPENAI_API_KEY'])

# csv 파일 벡터화 & 저장
from langchain_pinecone import PineconeVectorStore
BATCH_SIZE = 300 # chunk 개념으로 받아들이면 됨 -> 300개의 문장을 1개의 청크로 보면 됨
for i in range(0, len(docs), BATCH_SIZE):
    batch = docs[i: i + BATCH_SIZE]
    try:
        PineconeVectorStore.from_documents(
            documents = batch,
            index_name = os.environ['PINECONE_INDEX_NAME'],
            embedding = embedding
        )
        print(f'{i} ~ {i + len(batch) - 1} documents indexed')
    except Exception as e:
        print(f'{i} ~ {i + len(batch) - 1} error: {e}')
        