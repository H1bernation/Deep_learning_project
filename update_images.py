import pandas as pd
import psycopg2

# CSV 파일 읽기
df = pd.read_csv(r"C:\Users\황동민\Desktop\딥러닝\oliveyoung_raw_data\oliveyoung_products.csv")

print(f"총 {len(df)}개 제품 로드됨")
print(f"컬럼: {df.columns.tolist()}")
print(f"\n샘플 데이터:")
print(df[['name', 'image_url']].head())  # image → image_url

# DB 연결
conn = psycopg2.connect(
    host="localhost",
    port=5433,
    database="skincare_db",
    user="skincare_user",
    password="skincare_password"
)
cursor = conn.cursor()

# 업데이트
updated = 0
for idx, row in df.iterrows():
    product_name = row['name']
    image_url = row['image_url']  # image → image_url
    
    if pd.notna(image_url):
        cursor.execute(
            "UPDATE products SET image_url = %s WHERE name = %s",
            (image_url, product_name)
        )
        if cursor.rowcount > 0:
            updated += 1

conn.commit()
cursor.close()
conn.close()

print(f"\n✅ {updated}개 제품 이미지 업데이트 완료!")