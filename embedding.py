# embedding.py

from typing import List, Dict
from openai import OpenAI
import time


def _generate_embeddings_for_texts(texts: List[str], config: Dict) -> List[list[float] | None]:
    """
    一个通用的、核心的向量生成函数，接收一个字符串列表。
    """
    api_key = config.get('api_key')
    base_url = config.get('base_url')
    model_name = config.get('model_name')
    dimensions = config.get('dimensions')
    max_batch_size = config.get('max_batch_size', 25)

    if not all([api_key, base_url, model_name, dimensions]):
        print("⚠️  Warning: Embedding config is incomplete. Skipping embedding generation.")
        return [None] * len(texts)

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"❌ Error initializing OpenAI client: {e}. Skipping embedding generation.")
        return [None] * len(texts)

    all_embeddings = []
    print(f"🧬 Generating embeddings for {len(texts)} items using '{model_name}'...")

    for i in range(0, len(texts), max_batch_size):
        batch_texts = texts[i:i + max_batch_size]
        try:
            print(f"   - Processing batch {i // max_batch_size + 1}/{-(-len(texts) // max_batch_size)}...")
            response = client.embeddings.create(
                model=model_name,
                input=batch_texts,
                dimensions=dimensions
            )
            batch_embeddings = [embedding.embedding for embedding in response.data]
            all_embeddings.extend(batch_embeddings)
            time.sleep(0.1)  # 友好的请求间隔

        except Exception as e:
            print(f"❌ Error during API call for batch {i // max_batch_size + 1}: {e}")
            all_embeddings.extend([None] * len(batch_texts))

    return all_embeddings


def generate_chunk_embeddings(chunks: List[Dict], config: Dict) -> List[Dict]:
    """为文本块列表生成向量嵌入"""
    texts_to_embed = [chunk['text'] for chunk in chunks]
    embeddings = _generate_embeddings_for_texts(texts_to_embed, config)

    for chunk, embedding in zip(chunks, embeddings):
        chunk['embedding'] = embedding

    print("✅ Chunk embedding generation complete.")
    return chunks


def generate_entity_embeddings(entities: List[Dict], config: Dict) -> List[Dict]:
    """为实体列表生成向量嵌入"""
    # 仿照 lightrag 的格式，将名称和描述拼接为用于向量化的内容
    texts_to_embed = [f"{e['entity_name']}\n{e['description']}" for e in entities]
    embeddings = _generate_embeddings_for_texts(texts_to_embed, config)

    for entity, embedding in zip(entities, embeddings):
        entity['embedding'] = embedding

    print("✅ Entity embedding generation complete.")
    return entities


def generate_relation_embeddings(relations: List[Dict], config: Dict) -> List[Dict]:
    """为关系列表生成向量嵌入"""
    # 仿照 lightrag 的格式，拼接关键信息用于向量化
    texts_to_embed = [
        f"{r['keywords']}\t{r['source']}\n{r['target']}\n{r['description']}"
        for r in relations
    ]
    embeddings = _generate_embeddings_for_texts(texts_to_embed, config)

    for relation, embedding in zip(relations, embeddings):
        relation['embedding'] = embedding

    print("✅ Relation embedding generation complete.")
    return relations