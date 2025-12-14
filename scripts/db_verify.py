# db_verify.py
import psycopg2
from qdrant_client import QdrantClient

# 연결
pg_conn = psycopg2.connect(
    host="skincare-postgres",
    database="skincare_db",
    user="skincare_user",
    password="skincare_password",
    port=5432
)
pg_cursor = pg_conn.cursor()

qdrant_client = QdrantClient(host="skincare-qdrant", port=6333)

print("="*60)
print("데이터베이스 검증")
print("="*60)

# PostgreSQL 검증
print("\n[PostgreSQL]")
pg_cursor.execute("SELECT COUNT(*) FROM products;")
total = pg_cursor.fetchone()[0]
print(f"총 제품 수: {total}개")

pg_cursor.execute("""
    SELECT category, COUNT(*) 
    FROM products 
    GROUP BY category;
""")
print("\n카테고리별 분포:")
for row in pg_cursor.fetchall():
    print(f"  {row[0]}: {row[1]}개")

pg_cursor.execute("""
    SELECT price_tier, COUNT(*) 
    FROM products 
    GROUP BY price_tier;
""")
print("\n가격대별 분포:")
for row in pg_cursor.fetchall():
    print(f"  {row[0]}: {row[1]}개")

pg_cursor.execute("""
    SELECT 
        AVG(wrinkle_effect) as avg_wrinkle,
        AVG(pore_effect) as avg_pore,
        AVG(pigmentation_effect) as avg_pigment,
        AVG(sagging_effect) as avg_sag
    FROM products;
""")
row = pg_cursor.fetchone()
print("\n증상별 평균 효과:")
print(f"  주름: {row[0]:.2f}")
print(f"  모공: {row[1]:.2f}")
print(f"  색소: {row[2]:.2f}")
print(f"  처짐: {row[3]:.2f}")

pg_cursor.execute("""
    SELECT name, brand, price, wrinkle_effect, pore_effect 
    FROM products 
    WHERE wrinkle_effect >= 2 
    LIMIT 3;
""")
print("\n주름 효과 높은 제품 예시:")
for row in pg_cursor.fetchall():
    print(f"  {row[0]} ({row[1]}) - {row[2]:,}원 | 주름:{row[3]} 모공:{row[4]}")

# Qdrant 검증
print("\n[Qdrant]")
info = qdrant_client.get_collection("product_reviews")
print(f"벡터 수: {info.points_count}개")
print(f"벡터 차원: {info.config.params.vectors.size}")
print(f"거리 함수: {info.config.params.vectors.distance}")

# 샘플 검색 테스트
print("\n[검색 테스트]")
from openai import OpenAI
import os

# API 키가 환경변수에 없으면 스킵
if os.getenv('OPENAI_API_KEY'):
    client = OpenAI()
    test_query = "촉촉한 제품"
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=test_query
    ).data[0].embedding
    
    results = qdrant_client.search(
        collection_name="product_reviews",
        query_vector=embedding,
        limit=3
    )
    
    print(f"쿼리: '{test_query}'")
    print("상위 3개 결과:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.payload['name']} (유사도: {result.score:.4f})")
else:
    print("OpenAI API 키 없음 - 검색 테스트 스킵")

print("\n" + "="*60)
print("검증 완료")
print("="*60)

pg_cursor.close()
pg_conn.close()