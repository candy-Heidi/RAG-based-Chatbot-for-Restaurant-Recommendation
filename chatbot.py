#벡터 검색기능 최적화
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
import os
import math
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings


# MongoDB 서버에 연결
client = MongoClient(connection_string)
db = client['korea_rest']
cc_r = db['total_rest']  # 장소 데이터
cc_v = db['total_vector']  # 벡터 데이터

user_location = (33.430403, 126.927703)

# 거리 계산 함수
def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # 지구의 반지름 (단위: km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

# knn 함수 : 사용자의 위치에서 가까운 n개의 위치
def get_nearest_places(user_location, places, n):
    places_with_distance = [(place["id"], calculate_distance(user_location, place["location"])) for place in places]
    sorted_places = sorted(places_with_distance, key=lambda x: x[1])
    return [place for place, distance in sorted_places[:n]]

postcode3 = "636"  # 사용자 위치 -> postcode로 변환한 3자리 값

# 1차 필터링: postcode가 일치하는 장소
filtered_places = list(cc_r.find({"postcode": {"$regex": f"^{postcode3}"}}))

nearest_place_ids_int = get_nearest_places(user_location, filtered_places, n=50)
nearest_place_ids = [str(item) for item in nearest_place_ids_int]

# total_vector의 문서 구조에 맞춰 id 필드를 사용하여 조회
vector_docs = cc_v.find({"$or": [{str(nearest_id): {"$exists": True}} for nearest_id in nearest_place_ids]}, {'_id': 0})
small_db_MongoDB = {str(nearest_id): doc[str(nearest_id)]["embedding"] for doc in vector_docs for nearest_id in nearest_place_ids if str(nearest_id) in doc}


# 사용자 입력 임베딩 함수
embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'], model="text-embedding-3-small")

def embed_user_input(user_input):
    return embeddings.embed_query(user_input)

# 유사도 기반 검색
def retrieve_from_vector_database(user_input):
    k = 30
    user_embedding = embed_user_input(user_input)
    top_places = [None] * (k + 1)
    top_similarities = [float("-inf")] * (k + 1)
    most_similar_places = []

    for place_id, value in small_db_MongoDB.items():
        if value is not None:
            similarity = cosine_similarity([user_embedding], [value])[0][0]
            if similarity > min(top_similarities):
                min_similarity_index = top_similarities.index(min(top_similarities))
                top_places[min_similarity_index] = place_id
                top_similarities[min_similarity_index] = similarity

    for place_id in top_places:
        if place_id:
            similar_place = cc_r.find_one({"id": place_id})
            if similar_place:
                most_similar_places.append(similar_place)
    
    context = ""
    if most_similar_places:
        for rank, place in enumerate(most_similar_places, start=1):
            name = place.get("place_name", "Unknown")
            category = place.get("category_name", "Unknown")
            address = place.get("road_address_name", "Unknown")
            rating = place.get("rating", "Unknown")
            details = ", ".join(place.get("detail", []))
            context += f"k={rank}: 장소 이름은 '{name}'고 장소 타입은 '{category}'이다. 주소는 '{address}'이며 평점은 '{rating}'점이다. 세부사항으로는 '{details}' 등이 있다.\n"

    else:
        return "해당 장소 정보를 찾을 수 없습니다."

    context += f"\n사용자 입력: {user_input}"
    print(context)  # 최종 context 출력
    return context

# Langchain 설정
template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(temperature=0, api_key=os.environ['OPENAI_API_KEY'], model_name="gpt-3.5-turbo")

chain = (
    {"context": retrieve_from_vector_database, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 실행 및 결과 출력
user_input = "현재 위치 주변에서 별점 3점 이상이면서 애견 동반이 가능한 카페를 자세하게 알려줘"
result = chain.invoke(user_input)
print("질문 >>", user_input)
print(result)
