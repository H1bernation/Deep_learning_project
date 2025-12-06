from typing import Dict, List
import os
from openai import OpenAI
from qdrant_client import QdrantClient
import psycopg2

class RAGService:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant_client = QdrantClient(host="skincare-qdrant", port=6333)
        self.db_conn = psycopg2.connect(
            host="skincare-postgres",
            port=5432,
            database="skincare_db",
            user="skincare_user",
            password="skincare_password"
        )
    
    def get_product_details(self, product_ids: List[int]) -> List[Dict]:
        """추천된 제품의 상세 정보 가져오기"""
        cursor = self.db_conn.cursor()
        
        placeholders = ','.join(['%s'] * len(product_ids))
        query = f"""
            SELECT id, name, brand, price, category, 
                review_summary, avg_rating, review_count,
                wrinkle_effect, pore_effect, pigmentation_effect, sagging_effect
            FROM products
            WHERE id IN ({placeholders})
        """
        
        cursor.execute(query, product_ids)
        results = cursor.fetchall()
        
        products = []
        for row in results:
            products.append({
                'id': row[0],
                'name': row[1],
                'brand': row[2],
                'price': row[3],
                'category': row[4],
                'review_summary': row[5],
                'rating': row[6],  # avg_rating
                'review_count': row[7],
                'effects': {
                    'wrinkle': row[8],
                    'pore': row[9],
                    'pigmentation': row[10],
                    'sagging': row[11]
                }
            })
        
        cursor.close()
        return products
    def search_similar_reviews(self, query: str, top_k: int = 5) -> List[str]:
        """유사한 리뷰 검색"""
        # 쿼리 임베딩 생성
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_vector = response.data[0].embedding
        
        # Qdrant 검색
        search_results = self.qdrant_client.search(
            collection_name="product_reviews",
            query_vector=query_vector,
            limit=top_k
        )
        
        # 리뷰 텍스트 추출
        reviews = []
        for result in search_results:
            if hasattr(result, 'payload') and 'review_summary' in result.payload:
                reviews.append(result.payload['review_summary'])
        
        return reviews
    
    def generate_response(
        self, 
        question: str,
        analysis: Dict,
        recommendations: List[Dict],
        user_preferences: Dict = None
    ) -> str:
        """RAG 기반 답변 생성"""
        
        # 1. 제품 상세 정보 가져오기
        product_ids = [p['id'] for p in recommendations]
        product_details = self.get_product_details(product_ids)
        
        # 2. 유사 리뷰 검색
        similar_reviews = self.search_similar_reviews(question, top_k=3)
        
        # 3. 컨텍스트 구성
        context = f"""
# 피부 분석 결과
- 주름: {analysis['wrinkle']}/3
- 모공: {analysis['pore']}/3
- 색소침착: {analysis['pigmentation']}/3
- 처짐: {analysis['sagging']}/3

# 추천 제품 정보
"""
        for product in product_details:
            context += f"""
제품명: {product['name']}
브랜드: {product['brand']}
가격: {product['price']:,}원
카테고리: {product['category']}
평점: {product['rating']:.1f} ({product['review_count']}개 리뷰)
효과: 주름 {product['effects']['wrinkle']}, 모공 {product['effects']['pore']}, 색소 {product['effects']['pigmentation']}, 처짐 {product['effects']['sagging']}
리뷰 요약: {product['review_summary']}

"""
        
        if similar_reviews:
            context += "\n# 유사한 사용자 리뷰\n"
            for i, review in enumerate(similar_reviews, 1):
                context += f"{i}. {review}\n"
        
        if user_preferences:
            context += f"\n# 사용자 정보\n"
            context += f"피부 타입: {user_preferences.get('skin_type', '정보 없음')}\n"
            context += f"예산: {user_preferences.get('budget_min', 0):,}원 ~ {user_preferences.get('budget_max', 1000000):,}원\n"
        
        # 4. GPT-4o-mini 호출
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """당신은 전문 피부 관리 상담사입니다.
사용자의 피부 분석 결과와 추천 제품 정보를 바탕으로 친절하고 전문적인 답변을 제공합니다.

답변 가이드라인:
- 구체적인 제품 정보와 효과를 언급
- 실제 사용자 리뷰를 활용
- 피부 타입과 예산을 고려
- 간결하고 명확하게 (3-5문장)
- 이모지 사용 자제, 전문적 톤 유지"""
                },
                {
                    "role": "user",
                    "content": f"{context}\n\n질문: {question}"
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content