# embedding.py

from typing import List, Dict
from openai import OpenAI
import time


def _generate_embeddings_for_texts(texts: List[str], config: Dict) -> List[list[float] | None]:
    """
    一个通用的、核心的向量生成函数，接收一个字符串列表。
    新增了对无效文本（空或纯空白）的预过滤功能。
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

    # <--- 新增：预检查和清洗输入文本 ---
    cleaned_texts = []
    original_indices = []  # 记录有效文本的原始索引
    for i, text in enumerate(texts):
        # 如果文本不为空且不仅仅是空白字符，则认为是有效的
        if text and not text.isspace():
            cleaned_texts.append(text)
            original_indices.append(i)
        else:
            print(f"   - Skipping empty or whitespace-only chunk at original index {i}.")

    if not cleaned_texts:
        print("✅ No valid texts to embed in this batch.")
        return [None] * len(texts)  # 返回一个与原始输入等长的None列表

    # --- 后续流程仅处理清洗后的文本 ---
    all_embeddings = []
    print(f"🧬 Generating embeddings for {len(cleaned_texts)} valid chunks using '{model_name}'...")

    for i in range(0, len(cleaned_texts), max_batch_size):
        batch_texts = cleaned_texts[i:i + max_batch_size]
        try:
            print(f"   - Processing batch {i // max_batch_size + 1}/{-(-len(cleaned_texts) // max_batch_size)}...")
            response = client.embeddings.create(
                model=model_name,
                input=batch_texts,
                dimensions=dimensions
            )
            batch_embeddings = [embedding.embedding for embedding in response.data]
            all_embeddings.extend(batch_embeddings)
            time.sleep(0.1)

        except Exception as e:
            print(f"❌ Error during API call for batch {i // max_batch_size + 1}: {e}")
            all_embeddings.extend([None] * len(batch_texts))

    # <--- 新增：将生成的向量按原始索引映射回去 ---
    final_results = [None] * len(texts)  # 创建一个与原始输入等长的结果列表
    for i, embedding in enumerate(all_embeddings):
        # 获取当前向量对应的原始索引
        original_idx = original_indices[i]
        # 将向量放回正确的位置
        final_results[original_idx] = embedding

    return final_results


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
    texts_to_embed = [f"{e['entity_name']}\n{e['description']}" for e in entities]
    embeddings = _generate_embeddings_for_texts(texts_to_embed, config)

    for entity, embedding in zip(entities, embeddings):
        entity['embedding'] = embedding

    print("✅ Entity embedding generation complete.")
    return entities


def generate_relation_embeddings(relations: List[Dict], config: Dict) -> List[Dict]:
    """为关系列表生成向量嵌入"""
    # <--- MODIFIED: 使用新的列名 source_name 和 target_name ---
    texts_to_embed = [
        f"{r['keywords']}\t{r['source_name']}\n{r['target_name']}\n{r['description']}"
        for r in relations
    ]
    embeddings = _generate_embeddings_for_texts(texts_to_embed, config)

    for relation, embedding in zip(relations, embeddings):
        relation['embedding'] = embedding

    print("✅ Relation embedding generation complete.")
    return relations