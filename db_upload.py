# db_upload.py
import json
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from tqdm import tqdm
import time
import os
# 설정

PG_HOST = "skincare-postgres"
PG_DATABASE = "skincare_db"
PG_USER = "skincare_user"
PG_PASSWORD = "skincare_password"
PG_PORT = 5432

QDRANT_HOST = "skincare-qdrant"
QDRANT_PORT = 6333

JSON_FILE = "products_processed.json"

# 연결
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pg_conn = psycopg2.connect(
    host=PG_HOST,
    database=PG_DATABASE,
    user=PG_USER,
    password=PG_PASSWORD,
    port=PG_PORT
)
pg_cursor = pg_conn.cursor()
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

print("데이터베이스 연결 완료")

# PostgreSQL 테이블 생성
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

# Qdrant 컬렉션 생성
try:
    qdrant_client.delete_collection("product_reviews")
except:
    pass

qdrant_client.create_collection(
    collection_name="product_reviews",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

print("테이블 및 컬렉션 생성 완료")

# JSON 데이터 로드
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

products = data['products']
print(f"{len(products)}개 제품 로드")

# 임베딩 생성 함수
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# 데이터 업로드
print("\n데이터 업로드 시작")
successful = 0
failed = 0

for product in tqdm(products, desc="업로드 진행"):
    try:
        # PostgreSQL 삽입
        pg_cursor.execute("""
            INSERT INTO products (
                name, brand, category, price, price_tier, url,
                review_summary, review_count, avg_rating,
                wrinkle_effect, pore_effect, pigmentation_effect, sagging_effect,
                pros, cons, skin_types, age_groups
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            product['name'],
            product['brand'],
            product['category'],
            product['price'],
            product['price_tier'],
            product.get('url', ''),
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
        
        # Qdrant 삽입
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
        time.sleep(0.1)
        
    except Exception as e:
        failed += 1
        print(f"\n오류 발생: {product['name']} - {e}")
        pg_conn.rollback()

# 결과 확인
print(f"\n성공: {successful}개, 실패: {failed}개")

pg_cursor.execute("SELECT COUNT(*) FROM products;")
print(f"PostgreSQL: {pg_cursor.fetchone()[0]}개")

info = qdrant_client.get_collection("product_reviews")
print(f"Qdrant: {info.points_count}개")

pg_cursor.close()
pg_conn.close()