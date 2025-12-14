import json
import psycopg2
import pandas as pd
import os
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------
# [설정] 환경 변수 및 경로
# ---------------------------------------------------------
PG_HOST = "localhost"
PG_DATABASE = "skincare_db"
PG_USER = "skincare_user"
PG_PASSWORD = "skincare_password"
PG_PORT = 5433  # 외부 포트

QDRANT_HOST = "localhost"
QDRANT_PORT = 6334  # 외부 포트

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.join(BASE_DIR, "../data/oliveyoung/products_processed.json")
CSV_FILE = os.path.join(BASE_DIR, "../data/oliveyoung/oliveyoung_products.csv")


# [1] 데이터 로드 및 병합 (JSON + CSV)

print(f"[1/4] 데이터 로드 중...")

# 1. JSON 데이터 로드 (가공된 정보)
try:
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    products = data['products']
    print(f"   - JSON 로드 완료: {len(products)}개")
except FileNotFoundError:
    print(f"   [Error] JSON 파일을 찾을 수 없습니다: {JSON_FILE}")
    exit(1)

# 2. CSV 데이터 로드 (이미지 URL)
try:
    df = pd.read_csv(CSV_FILE)
    # 제품명을 키(Key), 이미지 URL을 값(Value)으로 하는 딕셔너리 생성
    image_map = dict(zip(df['name'], df['image_url']))
    print(f"   - CSV 로드 완료: {len(df)}개 (이미지 매핑 준비됨)")
except FileNotFoundError:
    print(f"   [Warning] CSV 파일을 찾을 수 없습니다. 이미지는 건너뜁니다.")
    image_map = {}


# [2] 데이터베이스 연결 및 초기화

print(f"[2/4] DB 연결 및 스키마 초기화...")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pg_conn = psycopg2.connect(
    host=PG_HOST, database=PG_DATABASE, user=PG_USER, password=PG_PASSWORD, port=PG_PORT
)
pg_cursor = pg_conn.cursor()
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# PostgreSQL 테이블 재생성 (image_url 컬럼 추가됨)
pg_cursor.execute("""
DROP TABLE IF EXISTS products CASCADE;

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    brand VARCHAR(200) NOT NULL,
    category VARCHAR(100) NOT NULL,
    price INTEGER NOT NULL,
    price_tier VARCHAR(20),
    url TEXT,
    image_url TEXT,  -- [New] 이미지 URL 컬럼 추가
    
    review_summary TEXT,
    review_count INTEGER,
    avg_rating FLOAT,
    
    wrinkle_effect INTEGER DEFAULT 0,
    pore_effect INTEGER DEFAULT 0,
    pigmentation_effect INTEGER DEFAULT 0,
    sagging_effect INTEGER DEFAULT 0,
    
    pros TEXT[],
    cons TEXT[],
    skin_types TEXT[],
    age_groups TEXT[],
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_category ON products(category);
CREATE INDEX idx_brand ON products(brand);
CREATE INDEX idx_price ON products(price);
CREATE INDEX idx_effects ON products(wrinkle_effect, pore_effect, pigmentation_effect, sagging_effect);
""")
pg_conn.commit()

# Qdrant 컬렉션 재생성
try:
    qdrant_client.delete_collection("product_reviews")
except:
    pass

qdrant_client.create_collection(
    collection_name="product_reviews",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
# [3] 데이터 업로드 (PostgreSQL + Qdrant)

print(f"[3/4] 데이터 업로드 시작...")

def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding

successful = 0
failed = 0

for product in tqdm(products, desc="Processing"):
    try:
        # 이미지 URL 매핑 (CSV에서 찾기)
        img_url = image_map.get(product['name'], None)
        
        # PostgreSQL 삽입
        pg_cursor.execute("""
            INSERT INTO products (
                name, brand, category, price, price_tier, url, image_url,
                review_summary, review_count, avg_rating,
                wrinkle_effect, pore_effect, pigmentation_effect, sagging_effect,
                pros, cons, skin_types, age_groups
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            product['name'],
            product['brand'],
            product['category'],
            product['price'],
            product['price_tier'],
            product.get('url', ''),
            img_url,  # 매핑된 이미지 URL 저장
            product.get('review_summary', ''),
            product.get('review_count', 0),
            product.get('avg_rating', 0.0),
            product['symptom_effects']['wrinkle'],
            product['symptom_effects']['pore'],
            product['symptom_effects']['pigmentation'],
            product['symptom_effects']['sagging'],
            product.get('pros', []),
            product.get('cons', []),
            product.get('skin_types', []),
            product.get('age_groups', [])
        ))
        
        product_id = pg_cursor.fetchone()[0]
        pg_conn.commit()
        
        # Qdrant 삽입 (리뷰가 있는 경우만)
        review_text = product.get('review_summary', '')
        if review_text:
            embedding = get_embedding(review_text)
            qdrant_client.upsert(
                collection_name="product_reviews",
                points=[
                    PointStruct(
                        id=product_id,
                        vector=embedding,
                        payload={
                            'product_id': product_id,
                            'name': product['name'],
                            'brand': product['brand'],
                            'category': product['category'],
                            'review_summary': review_text
                        }
                    )
                ]
            )
        
        successful += 1
        
    except Exception as e:
        failed += 1
        print(f"\n[Skip] {product['name']} - {e}")
        pg_conn.rollback()


# [4] 결과 확인

print(f"\n[완료] 성공: {successful}개, 실패: {failed}개")

pg_cursor.execute("SELECT COUNT(*) FROM products;")
pg_count = pg_cursor.fetchone()[0]
print(f" - PostgreSQL 저장됨: {pg_count}개")

qdrant_info = qdrant_client.get_collection("product_reviews")
print(f" - Qdrant 저장됨: {qdrant_info.points_count}개")

pg_cursor.close()
pg_conn.close()