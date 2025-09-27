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
    此版本为关系表增加了 source_id 和 target_id。
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
        # <--- MODIFIED: 更新关系表的列定义 ---
        global_relations_df = pd.DataFrame(
            columns=['relation_id', 'source_id', 'source_name', 'target_id', 'target_name', 'keywords', 'description',
                     'source_chunk_ids', 'degree', 'embedding'])

    try:
        global_chunks_df = pd.read_parquet(chunks_path)
    except FileNotFoundError:
        global_chunks_df = pd.DataFrame(columns=['chunk_id', 'text', 'token_count', 'embedding'])

    # --- 2. 融合实体 (逻辑不变) ---
    print(f"   - Fusing {len(all_new_entities)} new entities...")
    new_entities_df = pd.DataFrame(all_new_entities)
    combined_entities_df = pd.concat([global_entities_df.drop(columns=['embedding'], errors='ignore'), new_entities_df],
                                     ignore_index=True)

    if not combined_entities_df.empty:
        agg_funcs = {
            'entity_id': 'first',
            'entity_type': lambda x: Counter(x).most_common(1)[0][0],
            'description': lambda x: max(x, key=len),
            'source_chunk_ids': lambda x: list(set(sum(x.apply(list), []))),
            'degree': 'sum'
        }
        global_entities_df = combined_entities_df.groupby('entity_name').agg(agg_funcs).reset_index()
    else:
        global_entities_df = pd.DataFrame(columns=global_entities_df.columns.drop(['embedding'], errors='ignore'))

    # --- 3. 融合关系 ---
    print(f"   - Fusing {len(all_new_relations)} new relations...")
    new_relations_df = pd.DataFrame(all_new_relations)

    if not new_relations_df.empty or not global_relations_df.empty:
        # 重命名旧的 'source'/'target' 列（如果存在）以保持一致
        global_relations_df.rename(columns={'source': 'source_name', 'target': 'target_name'}, inplace=True)
        new_relations_df.rename(columns={'source': 'source_name', 'target': 'target_name'}, inplace=True)

        for df in [global_relations_df, new_relations_df]:
            if not df.empty:
                df['key'] = df.apply(lambda row: tuple(sorted((str(row['source_name']), str(row['target_name'])))),
                                     axis=1)

        combined_relations_df = pd.concat(
            [global_relations_df.drop(columns=['embedding'], errors='ignore'), new_relations_df], ignore_index=True)

        agg_funcs = {
            'relation_id': 'first',
            'source_name': 'first',
            'target_name': 'first',
            'keywords': lambda x: ", ".join(sorted(set(kw.strip() for kw_list in x for kw in kw_list.split(',')))),
            'description': lambda x: " | ".join(set(x)),
            'source_chunk_ids': lambda x: list(set(sum(x.apply(list), []))),
            'degree': 'sum'
        }

        global_relations_df = combined_relations_df.groupby('key').agg(agg_funcs).reset_index(drop=True)
    else:
        global_relations_df = pd.DataFrame(columns=global_relations_df.columns.drop(['embedding'], errors='ignore'))

    # --- 4. <NEW>: 映射实体ID到关系表 ---
    if not global_relations_df.empty and not global_entities_df.empty:
        print("   - Mapping entity IDs to relations...")
        # 创建一个从 entity_name 到 entity_id 的映射字典
        name_to_id_map = pd.Series(global_entities_df.entity_id.values, index=global_entities_df.entity_name).to_dict()

        # 使用 map 函数高效地添加 source_id 和 target_id
        global_relations_df['source_id'] = global_relations_df['source_name'].map(name_to_id_map)
        global_relations_df['target_id'] = global_relations_df['target_name'].map(name_to_id_map)

    # --- 5. 为最终的全局实体和关系生成向量 ---
    if embedding_config.get('api_key'):
        if not global_entities_df.empty:
            entities_list = global_entities_df.to_dict('records')
            entities_with_embeddings = generate_entity_embeddings(entities_list, embedding_config)
            global_entities_df = pd.DataFrame(entities_with_embeddings)

        if not global_relations_df.empty:
            relations_list = global_relations_df.to_dict('records')
            relations_with_embeddings = generate_relation_embeddings(relations_list, embedding_config)
            global_relations_df = pd.DataFrame(relations_with_embeddings)

    # --- 6. 保存融合后的全局实体和关系 ---
    if 'degree' in global_entities_df.columns:
        global_entities_df['degree'] = global_entities_df['degree'].fillna(0).astype(int)
    if 'degree' in global_relations_df.columns:
        global_relations_df['degree'] = global_relations_df['degree'].fillna(0).astype(int)

    # 重新排列关系表的列顺序
    if not global_relations_df.empty:
        final_relation_cols = [
            'relation_id', 'source_id', 'source_name', 'target_id', 'target_name',
            'keywords', 'description', 'source_chunk_ids', 'degree', 'embedding'
        ]
        # 确保所有列都存在，以防万一
        for col in final_relation_cols:
            if col not in global_relations_df.columns:
                global_relations_df[col] = None
        global_relations_df = global_relations_df[final_relation_cols]

    global_entities_df.to_parquet(entities_path, index=False)
    global_relations_df.to_parquet(relations_path, index=False)
    print(f"💾 Global entities with embeddings saved to: {entities_path}")
    print(f"💾 Global relations with embeddings saved to: {relations_path}")

    # --- 7 & 8. 更新和保存全局文本块 (逻辑不变) ---
    print(f"   - Updating global chunks with new data...")
    if 'embedding' not in global_chunks_df.columns:
        global_chunks_df['embedding'] = None
    global_chunks_df = pd.concat([global_chunks_df, all_new_chunks_df]).drop_duplicates(subset=['chunk_id'],
                                                                                        keep='last').reset_index(
        drop=True)

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

    global_chunks_df.to_parquet(chunks_path, index=False)
    print(f"💾 Global chunks with KG mapping saved to: {chunks_path}")

    print("✅ Global knowledge fusion finished.")