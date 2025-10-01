# fusion.py

import pandas as pd
from pathlib import Path
from typing import List, Dict
from embedding import generate_entity_embeddings, generate_relation_embeddings
from collections import Counter, defaultdict

from extraction import _get_unique_id


def fuse_and_update_knowledge_base(
        all_new_entities: List[Dict],
        all_new_relations: List[Dict],
        all_new_chunks_df: pd.DataFrame,
        rag_space_path: Path,
        embedding_config: Dict,
        token_usage: Dict
):
    """
    执行全局知识融合，并修复因LLM提取不一致导致的ID映射为null的问题。
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
                     'frequency', 'embedding'])
    try:
        global_relations_df = pd.read_parquet(relations_path)
    except FileNotFoundError:
        global_relations_df = pd.DataFrame(
            columns=['relation_id', 'source_id', 'source_name', 'target_id', 'target_name', 'keywords', 'description',
                     'source_chunk_ids', 'frequency', 'degree', 'embedding'])
    try:
        global_chunks_df = pd.read_parquet(chunks_path)
    except FileNotFoundError:
        global_chunks_df = pd.DataFrame(
            columns=['chunk_id', 'text', 'token_count', 'embedding', 'source_document_name'])

    # --- 2. 融合实体 ---
    print(f"   - Fusing {len(all_new_entities)} new entities...")
    new_entities_df = pd.DataFrame(all_new_entities)
    if not new_entities_df.empty:
        new_entities_df['entity_name'] = new_entities_df['entity_name'].str.strip()

    combined_entities_df = pd.concat([global_entities_df, new_entities_df], ignore_index=True)

    if not combined_entities_df.empty:
        old_embeddings = combined_entities_df[['entity_id', 'embedding']].dropna(subset=['embedding']).drop_duplicates(
            subset=['entity_id'], keep='first')
        agg_funcs = {
            'entity_id': 'first', 'entity_type': lambda x: Counter(x).most_common(1)[0][0],
            'description': lambda x: max(x, key=len),
            'source_chunk_ids': lambda x: list(set(sum(x.apply(list), []))),
            'frequency': 'sum'
        }
        final_entities_df = combined_entities_df.groupby('entity_name', as_index=False).agg(agg_funcs)
        final_entities_df = final_entities_df.merge(old_embeddings, on='entity_id', how='left')
    else:
        final_entities_df = pd.DataFrame(columns=global_entities_df.columns)

    # --- 3. 融合关系 ---
    print(f"   - Fusing {len(all_new_relations)} new relations...")
    new_relations_df = pd.DataFrame(all_new_relations)
    if not new_relations_df.empty:
        new_relations_df.rename(columns={'source': 'source_name', 'target': 'target_name'}, inplace=True)
        new_relations_df['source_name'] = new_relations_df['source_name'].str.strip()
        new_relations_df['target_name'] = new_relations_df['target_name'].str.strip()

    if not new_relations_df.empty or not global_relations_df.empty:
        if 'source' in global_relations_df.columns:
            global_relations_df.rename(columns={'source': 'source_name', 'target': 'target_name'}, inplace=True)

        for df in [global_relations_df, new_relations_df]:
            if not df.empty and 'source_name' in df.columns:
                df['source_name'] = df['source_name'].str.strip()
                df['target_name'] = df['target_name'].str.strip()
                df['key'] = df.apply(lambda row: tuple(sorted((str(row['source_name']), str(row['target_name'])))),
                                     axis=1)

        old_relations_embeddings = global_relations_df[['relation_id', 'embedding']].dropna(
            subset=['embedding']).drop_duplicates(subset=['relation_id'], keep='first')

        combined_relations_df = pd.concat([global_relations_df, new_relations_df], ignore_index=True)

        agg_funcs = {
            'relation_id': 'first', 'source_name': 'first', 'target_name': 'first',
            'keywords': lambda x: ", ".join(
                sorted(set(kw.strip() for kw_list in x if kw_list for kw in str(kw_list).split(',')))),
            'description': lambda x: " | ".join(set(x)),
            'source_chunk_ids': lambda x: list(set(sum(x.apply(list), []))),
            'frequency': 'sum'
        }

        final_relations_df = combined_relations_df.groupby('key').agg(agg_funcs).reset_index(drop=True)
        final_relations_df = final_relations_df.merge(old_relations_embeddings, on='relation_id', how='left')
    else:
        final_relations_df = pd.DataFrame(columns=global_relations_df.columns)

    # --- 4. 实体补全机制 ---
    if not final_relations_df.empty:
        print("   - Completing missing entities from relations...")
        relation_entity_names = set(final_relations_df['source_name']).union(set(final_relations_df['target_name']))
        existing_entity_names = set(final_entities_df['entity_name'])

        missing_names = relation_entity_names - existing_entity_names
        if missing_names:
            print(f"     - Found {len(missing_names)} missing entities. Creating placeholders...")
            new_placeholder_entities = []
            for name in missing_names:
                new_placeholder_entities.append({
                    "entity_id": _get_unique_id(name, prefix="ent-"),
                    "entity_name": name,
                    "entity_type": "UNKNOWN",
                    "description": "",  # <--- MODIFIED: 将描述设置为空字符串
                    "source_chunk_ids": [],
                    "frequency": 0,
                    "degree": 0
                })

            missing_entities_df = pd.DataFrame(new_placeholder_entities)
            final_entities_df = pd.concat([final_entities_df, missing_entities_df], ignore_index=True)

    # --- 5. ID 映射 ---
    if not final_relations_df.empty and not final_entities_df.empty:
        print("   - Mapping entity IDs to relations...")
        name_to_id_map = pd.Series(final_entities_df.entity_id.values, index=final_entities_df.entity_name).to_dict()
        final_relations_df['source_id'] = final_relations_df['source_name'].map(name_to_id_map)
        final_relations_df['target_id'] = final_relations_df['target_name'].map(name_to_id_map)

    # --- 6. 计算实体连接度 (degree) ---
    if not final_entities_df.empty and not final_relations_df.empty:
        print("   - Calculating entity degrees (connectivity)...")
        degree_counter = defaultdict(int)
        clean_relations_for_degree = final_relations_df.dropna(subset=['source_id', 'target_id'])
        for _, row in clean_relations_for_degree.iterrows():
            degree_counter[row['source_id']] += 1
            degree_counter[row['target_id']] += 1
        final_entities_df['degree'] = final_entities_df['entity_id'].map(degree_counter).fillna(0).astype(int)

    # --- 7. 为关系表增加 degree (等于 frequency) ---
    if not final_relations_df.empty:
        final_relations_df['degree'] = final_relations_df['frequency']

    # --- 8. 增量向量化 ---
    if embedding_config.get('api_key'):
        if not final_entities_df.empty:
            entities_to_embed_df = final_entities_df[final_entities_df['embedding'].isnull()]
            if not entities_to_embed_df.empty:
                print(f"   - Generating embeddings for {len(entities_to_embed_df)} new/updated entities...")
                entities_list = entities_to_embed_df.to_dict('records')
                entities_with_embeddings, tokens_used = generate_entity_embeddings(entities_list, embedding_config)
                token_usage["embedding_entities"] += tokens_used
                print(f"   - Tokens used for entities embedding: {tokens_used}")

                embed_df = pd.DataFrame(entities_with_embeddings).set_index('entity_id')
                final_entities_df = final_entities_df.set_index('entity_id')
                final_entities_df.update(embed_df)
                final_entities_df.reset_index(inplace=True)

        if not final_relations_df.empty:
            relations_to_embed_df = final_relations_df[final_relations_df['embedding'].isnull()]
            if not relations_to_embed_df.empty:
                print(f"   - Generating embeddings for {len(relations_to_embed_df)} new/updated relations...")
                relations_list = relations_to_embed_df.to_dict('records')
                relations_with_embeddings, tokens_used = generate_relation_embeddings(relations_list, embedding_config)
                token_usage["embedding_relations"] += tokens_used
                print(f"   - Tokens used for relations embedding: {tokens_used}")

                embed_df = pd.DataFrame(relations_with_embeddings).set_index('relation_id')
                final_relations_df = final_relations_df.set_index('relation_id')
                final_relations_df.update(embed_df)
                final_relations_df.reset_index(inplace=True)

    # --- 9. 保存最终的全局文件 ---
    # ... (后续代码与上一版完全相同，此处省略以保持简洁) ...
    if 'degree' in final_entities_df.columns:
        final_entities_df['degree'] = final_entities_df['degree'].fillna(0).astype(int)
    if 'frequency' in final_entities_df.columns:
        final_entities_df['frequency'] = final_entities_df['frequency'].fillna(0).astype(int)
    if 'degree' in final_relations_df.columns:
        final_relations_df['degree'] = final_relations_df['degree'].fillna(0).astype(int)
    if 'frequency' in final_relations_df.columns:
        final_relations_df['frequency'] = final_relations_df['frequency'].fillna(0).astype(int)

    if not final_relations_df.empty:
        final_relation_cols = ['relation_id', 'source_id', 'source_name', 'target_id', 'target_name', 'keywords',
                               'description', 'source_chunk_ids', 'frequency', 'degree', 'embedding']
        for col in final_relation_cols:
            if col not in final_relations_df.columns:
                final_relations_df[col] = None
        if 'key' in final_relations_df.columns:
            final_relations_df = final_relations_df.drop(columns=['key'])
        final_relations_df = final_relations_df[final_relation_cols]

    final_entities_df.to_parquet(entities_path, index=False)
    final_relations_df.to_parquet(relations_path, index=False)
    print(f"💾 Global entities with embeddings saved to: {entities_path}")
    print(f"💾 Global relations with embeddings saved to: {relations_path}")

    # --- 10 & 11. 更新和保存全局文本块 ---
    print(f"   - Updating global chunks with new data...")
    if 'source_document_name' not in global_chunks_df.columns:
        global_chunks_df['source_document_name'] = None
    if 'embedding' not in global_chunks_df.columns:
        global_chunks_df['embedding'] = None
    global_chunks_df = pd.concat([global_chunks_df, all_new_chunks_df]).drop_duplicates(subset=['chunk_id'],
                                                                                        keep='last').reset_index(
        drop=True)

    print(f"   - Mapping knowledge graph back to chunks...")
    if 'entity_ids' not in global_chunks_df.columns:
        global_chunks_df['entity_ids'] = pd.Series([[] for _ in range(len(global_chunks_df))])
    if 'relation_ids' not in global_chunks_df.columns:
        global_chunks_df['relation_ids'] = pd.Series([[] for _ in range(len(global_chunks_df))])

    chunk_map = {chunk_id: {'entity_ids': [], 'relation_ids': []} for chunk_id in global_chunks_df['chunk_id']}

    for _, entity in final_entities_df.iterrows():
        sources = entity.get('source_chunk_ids', [])
        if isinstance(sources, list):
            for chunk_id in sources:
                if chunk_id in chunk_map:
                    chunk_map[chunk_id]['entity_ids'].append(entity['entity_id'])

    for _, relation in final_relations_df.iterrows():
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