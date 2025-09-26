# fusion.py

import pandas as pd
from pathlib import Path
from typing import List, Dict
from embedding import generate_entity_embeddings, generate_relation_embeddings
from collections import Counter


def fuse_and_update_knowledge_base(
        all_new_entities: List[Dict],
        all_new_relations: List[Dict],
        all_new_chunks_df: pd.DataFrame,
        rag_space_path: Path,
        embedding_config: Dict
):
    """
    执行全局知识融合，为实体和关系生成向量，并更新所有 Parquet 文件。
    此版本包含最终的健壮合并逻辑，以处理各种数据重复情况。
    """
    print("\n🔗 Starting global knowledge fusion...")
    rag_space_path.mkdir(exist_ok=True)

    entities_path = rag_space_path / "vdb_entities.parquet"
    relations_path = rag_space_path / "vdb_relationships.parquet"
    chunks_path = rag_space_path / "vdb_chunks.parquet"

    # --- 1. 加载或创建全局数据 ---
    try:
        global_entities_df = pd.read_parquet(entities_path)
    except FileNotFoundError:
        global_entities_df = pd.DataFrame(
            columns=['entity_id', 'entity_name', 'entity_type', 'description', 'source_chunk_ids', 'degree',
                     'embedding'])

    try:
        global_relations_df = pd.read_parquet(relations_path)
    except FileNotFoundError:
        global_relations_df = pd.DataFrame(
            columns=['relation_id', 'source', 'target', 'keywords', 'description', 'source_chunk_ids', 'degree',
                     'embedding'])

    try:
        global_chunks_df = pd.read_parquet(chunks_path)
    except FileNotFoundError:
        global_chunks_df = pd.DataFrame(columns=['chunk_id', 'text', 'token_count', 'embedding'])

    # --- 2. 融合实体 ---
    print(f"   - Fusing {len(all_new_entities)} new entities...")
    new_entities_df = pd.DataFrame(all_new_entities)

    # 合并新旧实体
    combined_entities_df = pd.concat([global_entities_df, new_entities_df], ignore_index=True)

    if not combined_entities_df.empty:
        # 定义聚合函数
        agg_funcs = {
            'entity_id': 'first',
            'entity_type': lambda x: Counter(x).most_common(1)[0][0],
            'description': lambda x: max(x, key=len),
            # <--- MODIFIED: 在 sum 之前，使用 apply(list) 强制转换类型 ---
            'source_chunk_ids': lambda x: list(set(sum(x.apply(list), []))),
            'degree': 'sum'
        }

        global_entities_df = combined_entities_df.groupby('entity_name').agg(agg_funcs).reset_index()
    else:
        global_entities_df = pd.DataFrame(columns=global_entities_df.columns)

    # --- 3. 融合关系 ---
    print(f"   - Fusing {len(all_new_relations)} new relations...")
    new_relations_df = pd.DataFrame(all_new_relations)

    if not new_relations_df.empty or not global_relations_df.empty:
        for df in [global_relations_df, new_relations_df]:
            if not df.empty:
                df['key'] = df.apply(lambda row: tuple(sorted((str(row['source']), str(row['target'])))), axis=1)

        combined_relations_df = pd.concat([global_relations_df, new_relations_df], ignore_index=True)

        agg_funcs = {
            'relation_id': 'first',
            'source': 'first',
            'target': 'first',
            'keywords': lambda x: ", ".join(sorted(set(kw.strip() for kw_list in x for kw in kw_list.split(',')))),
            'description': lambda x: " | ".join(set(x)),
            # <--- MODIFIED: 在 sum 之前，使用 apply(list) 强制转换类型 ---
            'source_chunk_ids': lambda x: list(set(sum(x.apply(list), []))),
            'degree': 'sum'
        }

        global_relations_df = combined_relations_df.groupby('key').agg(agg_funcs).reset_index(drop=True)
    else:
        global_relations_df = pd.DataFrame(columns=global_relations_df.columns)

    if 'key' in global_relations_df.columns:
        global_relations_df = global_relations_df.drop(columns=['key'])

    # --- 4. 为最终的全局实体和关系生成向量 ---
    if embedding_config.get('api_key'):
        # 仅对发生变化或新增的行进行向量化，以提高效率
        # (为简化，这里仍然对全部数据进行向量化，未来可优化)
        if not global_entities_df.empty:
            entities_list = global_entities_df.to_dict('records')
            entities_with_embeddings = generate_entity_embeddings(entities_list, embedding_config)
            global_entities_df = pd.DataFrame(entities_with_embeddings)

        if not global_relations_df.empty:
            relations_list = global_relations_df.to_dict('records')
            relations_with_embeddings = generate_relation_embeddings(relations_list, embedding_config)
            global_relations_df = pd.DataFrame(relations_with_embeddings)

    # --- 5. 保存融合后的全局实体和关系 ---
    if 'degree' in global_entities_df.columns:
        global_entities_df['degree'] = global_entities_df['degree'].fillna(0).astype(int)
    if 'degree' in global_relations_df.columns:
        global_relations_df['degree'] = global_relations_df['degree'].fillna(0).astype(int)

    global_entities_df.to_parquet(entities_path, index=False)
    global_relations_df.to_parquet(relations_path, index=False)
    print(f"💾 Global entities with embeddings saved to: {entities_path}")
    print(f"💾 Global relations with embeddings saved to: {relations_path}")

    # --- 6. 更新全局文本块 ---
    print(f"   - Updating global chunks with new data...")
    if 'embedding' not in global_chunks_df.columns:
        global_chunks_df['embedding'] = None
    global_chunks_df = pd.concat([global_chunks_df, all_new_chunks_df]).drop_duplicates(subset=['chunk_id'],
                                                                                        keep='last').reset_index(
        drop=True)

    # --- 7. 映射实体和关系ID回文本块 ---
    print(f"   - Mapping knowledge graph back to chunks...")
    # ... 映射逻辑保持不变 ...
    if 'entity_ids' not in global_chunks_df.columns:
        global_chunks_df['entity_ids'] = pd.Series([[] for _ in range(len(global_chunks_df))])
    if 'relation_ids' not in global_chunks_df.columns:
        global_chunks_df['relation_ids'] = pd.Series([[] for _ in range(len(global_chunks_df))])

    chunk_map = {chunk_id: {'entity_ids': [], 'relation_ids': []} for chunk_id in global_chunks_df['chunk_id']}

    for _, entity in global_entities_df.iterrows():
        sources = entity.get('source_chunk_ids', [])
        if isinstance(sources, list):
            for chunk_id in sources:
                if chunk_id in chunk_map:
                    chunk_map[chunk_id]['entity_ids'].append(entity['entity_id'])

    for _, relation in global_relations_df.iterrows():
        sources = relation.get('source_chunk_ids', [])
        if isinstance(sources, list):
            for chunk_id in sources:
                if chunk_id in chunk_map:
                    chunk_map[chunk_id]['relation_ids'].append(relation['relation_id'])

    map_df = pd.DataFrame.from_dict(chunk_map, orient='index').reset_index().rename(columns={'index': 'chunk_id'})

    if 'entity_ids' in global_chunks_df.columns:
        global_chunks_df = global_chunks_df.drop(columns=['entity_ids'])
    if 'relation_ids' in global_chunks_df.columns:
        global_chunks_df = global_chunks_df.drop(columns=['relation_ids'])

    global_chunks_df = global_chunks_df.merge(map_df, on='chunk_id', how='left')

    # --- 8. 保存最终的全局文本块文件 ---
    global_chunks_df.to_parquet(chunks_path, index=False)
    print(f"💾 Global chunks with KG mapping saved to: {chunks_path}")

    print("✅ Global knowledge fusion finished.")