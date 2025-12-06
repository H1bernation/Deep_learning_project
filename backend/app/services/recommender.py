import psycopg2
from qdrant_client import QdrantClient
from openai import OpenAI
from typing import Dict, List
import math
import os

class RecommendationEngine:
    def __init__(self):
        self.pg_conn = psycopg2.connect(
            host="skincare-postgres",
            database="skincare_db",
            user="skincare_user",
            password="skincare_password",
            port=5432
        )
        self.qdrant = QdrantClient(host="skincare-qdrant", port=6333)
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def recommend(self, skin_analysis: Dict, user_preferences: Dict, query: str = None) -> Dict:
        # Stage 1: Hard Filtering
        candidates = self._knowledge_filter(
            user_preferences['skin_type'],
            user_preferences['budget_min'],
            user_preferences['budget_max'],
            user_preferences.get('category')
        )
        
        print(f"Stage 1 필터링: {len(candidates)}개")
        
        # Stage 2: Symptom Score
        for product in candidates:
            product['symptom_score'] = self._calculate_symptom_score(skin_analysis, product)
        
        # Stage 3: Semantic Search (query 있을 때만)
        if query:
            semantic_results = self._semantic_search(query, candidates)
            score_map = {r.id: r.score for r in semantic_results}
            for product in candidates:
                product['semantic_score'] = score_map.get(product['id'], 0.5)
        else:
            for product in candidates:
                product['semantic_score'] = 0.5
        
        # Final: 종합 점수 계산 및 정렬
        for product in candidates:
            product['score'] = self._calculate_final_score(product)
            product['explanation'] = self._generate_explanation(
                product, 
                skin_analysis, 
                user_preferences
            )

        # 카테고리 지정 안 했으면 다양성 적용
        if not user_preferences.get('category'):
            top_products = self._ensure_category_diversity(candidates)
        else:
            top_products = sorted(
                candidates, 
                key=lambda x: x['score'],
                reverse=True
            )[:9]
        
        return {
            'products': top_products,
            'total_candidates': len(candidates)
        }
    
    def _knowledge_filter(self, skin_type, budget_min, budget_max, category):
        cursor = self.pg_conn.cursor()
        

        query = """
            SELECT id, name, brand, category, price,
                   wrinkle_effect, pore_effect, pigmentation_effect, sagging_effect,
                   review_summary, avg_rating, review_count,
                   url, image_url 
            FROM products
            WHERE %s = ANY(skin_types)
              AND price BETWEEN %s AND %s
        """
        params = [skin_type, budget_min, budget_max]
        
        if category:
            query += " AND category = %s"
            params.append(category)
        
        cursor.execute(query, params)
        
        products = []
        for row in cursor.fetchall():
            products.append({
                'id': row[0], 'name': row[1], 'brand': row[2],
                'category': row[3], 'price': row[4],
                'effects': {
                    'wrinkle': row[5], 'pore': row[6],
                    'pigmentation': row[7], 'sagging': row[8]
                },
                'review_summary': row[9],
                'avg_rating': float(row[10]) if row[10] else 0.0,
                'review_count': row[11],
                'url': row[12],
                'image': row[13] 
            })
        
        cursor.close()
        return products
    
    def _calculate_symptom_score(self, user_symptoms, product):
        symptoms = ['wrinkle', 'pore', 'pigmentation', 'sagging']
        weighted_sum = sum(user_symptoms[s] * product['effects'][s] for s in symptoms)
        weight_total = sum(user_symptoms[s] for s in symptoms)
        
        if weight_total == 0:
            return 0.5
        
        return min(weighted_sum / (weight_total * 3), 1.0)
    
    def _semantic_search(self, query, candidates):
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        
        embedding = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding
        
        results = self.qdrant.query_points(
            collection_name="product_reviews",
            query=embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="product_id",
                        match=MatchAny(any=[p['id'] for p in candidates])
                    )
                ]
            ),
            limit=len(candidates)
        )
        
        return results.points
    
    def _calculate_final_score(self, product):
        review_score = (
            (product['avg_rating'] / 5.0) * 0.7 +
            min(math.log(product['review_count'] + 1) / math.log(100), 1.0) * 0.3
        )
        
        return round(
            0.40 * product['symptom_score'] +
            0.30 * product['semantic_score'] +
            0.20 * review_score +
            0.10 * 1.0,
            4
        )
    
    def _generate_explanation(self, product, user_symptoms, user_preferences):
        reasons = []
        symptom_kr = {
            'wrinkle': '주름', 'pore': '모공',
            'pigmentation': '색소', 'sagging': '처짐'
        }
        
        for s in ['wrinkle', 'pore', 'pigmentation', 'sagging']:
            if user_symptoms[s] >= 1 and product['effects'][s] >= 2:
                if product['effects'][s] == 3:
                    reasons.append(f"강력 {symptom_kr[s]} 케어")
                else:
                    reasons.append(f"{symptom_kr[s]} 개선")
        
        if not reasons:
            skin_type = user_preferences.get('skin_type', '')
            if skin_type:
                reasons.append(f"{skin_type} 피부 적합")
        
        if product['avg_rating'] >= 4.5:
            reasons.append(f"평점 {product['avg_rating']:.1f}")
        
        if product['review_count'] >= 80:
            reasons.append("다수 리뷰")
        
        return " | ".join(reasons) if reasons else "종합 추천"
    
    def _ensure_category_diversity(self, candidates, top_k=9):
        category_keywords = {
            '토너': ['토너', '스킨'],
            '로션': ['로션', '에멀젼', '크림'],
            '세럼': ['에센스', '세럼', '앰플']
        }
        
        grouped = {'토너': [], '로션': [], '세럼': []}
        ungrouped = []
        
        for product in candidates:
            cat = product['category']
            matched = False
            
            for group_name, keywords in category_keywords.items():
                if any(keyword in cat for keyword in keywords):
                    grouped[group_name].append(product)
                    matched = True
                    break
            
            if not matched:
                ungrouped.append(product)
        
        for group in grouped.values():
            group.sort(key=lambda x: x['score'], reverse=True)
        
        result = []
        for group_name in ['토너', '로션', '세럼']:
            result.extend(grouped[group_name][:3])
        
        if len(result) < top_k:
            used_ids = {p['id'] for p in result}
            remaining = [
                p for p in candidates 
                if p['id'] not in used_ids
            ]
            remaining.sort(key=lambda x: x['score'], reverse=True)
            result.extend(remaining[:top_k - len(result)])
        
        result.sort(key=lambda x: x['score'], reverse=True)
        
        return result[:top_k]