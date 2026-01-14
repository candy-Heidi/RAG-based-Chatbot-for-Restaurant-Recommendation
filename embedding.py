import os
import time
from langchain_openai import OpenAIEmbeddings
import json

embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'], model="text-embedding-ada-002")

with open('rest_jeju.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

vector_data = {}

for idx, desc in enumerate(data):
    try:
        latitude, longitude = desc["location"]
        postcode = desc["postcode"]
        phone = desc["phone"]
        place_name = desc["place_name"]
        category_name = desc["category_name"]
        road_address_name = desc["road_address_name"]
        rating = str(desc["rating"])
        distance = str(desc["distance"])
        detail = " ".join(desc.get("detail", []))
        page_content = f"장소 이름은 {place_name}이고 장소의 타입은 {category_name}입니다. 해당 장소의 평점은 {rating}점이고, 현 위치와의 거리는 {distance}입니다. 세부 특징으로는 {detail} 등이 있습니다."

        print(f"Processing {idx + 1}/{len(data)}: {place_name}")  # 로그 추가

        embedding = embeddings.embed_query(page_content)
        vector_data[desc["id"]] = {
            "page_content": page_content,
            "embedding": embedding  
        }

        time.sleep(1)  # API 호출 사이에 대기 시간을 추가하여 속도 제한 완화
    except Exception as e:
        print(f"Error processing {desc['id']}: {e}")

with open('vector_seouljeju.json', 'w') as file:
    json.dump(vector_data, file)
