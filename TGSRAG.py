import sys
from pathlib import Path
import hashlib
import json
import yaml
import pandas as pd
import asyncio

from pdf2md import process_pdf
from chunks import chunk_dispatcher
from embedding import generate_chunk_embeddings
from extraction import extract_entities_and_relations
from fusion import fuse_and_update_knowledge_base


def main():
    print("🚀 Starting BiRAG processing pipeline...")

    # 初始化总token计数器
    total_token_usage = {
        "embedding_chunks": 0,
        "extraction": 0,
        "embedding_entities": 0,
        "embedding_relations": 0
    }

    input_dir = Path("D:/SEU_study/AI4MW/Text-Graph-Synergy-RAG/input14_3")
    output_dir_base = Path("D:/SEU_study/AI4MW/Text-Graph-Synergy-RAG/rag_storage_hotpotqa")
    config_path = Path("config.yaml")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ 配置文件 '{config_path}' 未找到。请确保文件存在。")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 加载配置文件时出错: {e}。")
        sys.exit(1)

    general_config = config.get('General', {})
    chunking_config = config.get('Chunking', {})
    embedding_config = config.get('Embedding', {})
    llm_config = config.get('LLM', {})
    extraction_config = config.get('Extraction', {})

    rag_space = general_config.get('rag_space')
    if not rag_space:
        print("❌ 错误: 'rag_space' 未在 config.yaml 中配置。")
        sys.exit(1)

    rag_space_path = Path(rag_space)
    input_dir.mkdir(exist_ok=True)
    output_dir_base.mkdir(exist_ok=True)

    # <--- DELETED: 移除了用于批处理的全局列表 ---

    for doc_path in input_dir.iterdir():
        if not doc_path.is_file():
            continue

        print(f"\n{'=' * 50}\n-> Found document: {doc_path.name}")
        doc_hash_name = hashlib.md5(doc_path.read_bytes()).hexdigest()
        unique_output_dir = output_dir_base / doc_hash_name

        if unique_output_dir.exists():
            print(f"✅ Document '{doc_path.name}' (hash: {doc_hash_name}) has been processed before. Skipping.")
            continue

        unique_output_dir.mkdir(parents=True, exist_ok=True)

        text_to_chunk = ""
        if doc_path.suffix.lower() == '.pdf':
            md_path = process_pdf(doc_path, unique_output_dir)
            if md_path and md_path.is_file():
                text_to_chunk = md_path.read_text(encoding='utf-8')
        elif doc_path.suffix.lower() in ['.txt', '.md']:
            text_to_chunk = doc_path.read_text(encoding='utf-8')
        else:
            print(f"ℹ️  Skipping: Unsupported file type for '{doc_path.name}'")
            continue

        if text_to_chunk:
            # 1. 分块
            print("🧠 Chunking text...")
            chunks = chunk_dispatcher(text_to_chunk, doc_hash_name, chunking_config)

            source_filename = doc_path.name
            for chunk in chunks:
                chunk['source_document_name'] = source_filename

            chunks_dir = unique_output_dir / "chunks"
            chunks_dir.mkdir(exist_ok=True)
            with open(chunks_dir / "chunks.json", 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)

            successful_chunks = []
            if embedding_config.get('api_key'):
                # 2. 生成向量
                chunks_with_embeddings, tokens_used = generate_chunk_embeddings(chunks, embedding_config)
                total_token_usage["embedding_chunks"] += tokens_used
                print(f"   - Tokens used for chunks embedding: {tokens_used}")

                successful_chunks = [c for c in chunks_with_embeddings if c.get('embedding') is not None]

                if successful_chunks:
                    df_chunks = pd.DataFrame(successful_chunks)
                    cols = ['chunk_id', 'source_document_name', 'text', 'token_count', 'embedding']
                    df_chunks = df_chunks[[col for col in cols if col in df_chunks.columns]]
                    parquet_path = chunks_dir / "vdb_chunks.parquet"
                    df_chunks.to_parquet(parquet_path, index=False)
                    print(f"💾 Per-document vector data saved to: {parquet_path}")
                else:
                    print("⚠️ No embeddings were successfully generated for this document.")
            else:
                print("ℹ️  Skipping embedding generation because API key is not configured.")
                successful_chunks = chunks

            if llm_config.get('api_key') and successful_chunks:
                # 3. 抽取实体和关系
                print("✨ Starting entity and relationship extraction for this document...")
                doc_entities, doc_relations, tokens_used = asyncio.run(extract_entities_and_relations(
                    successful_chunks, llm_config, extraction_config
                ))
                total_token_usage["extraction"] += tokens_used
                print(f"   - Tokens used for extraction: {tokens_used}")

                kg_dir = unique_output_dir / "kg"
                kg_dir.mkdir(exist_ok=True)
                with open(kg_dir / "entities.json", 'w', encoding='utf-8') as f:
                    json.dump(doc_entities, f, indent=2, ensure_ascii=False)
                with open(kg_dir / "relations.json", 'w', encoding='utf-8') as f:
                    json.dump(doc_relations, f, indent=2, ensure_ascii=False)
                print(f"💾 Raw extraction results saved to: {kg_dir}")

                # <--- MODIFIED: 将全局融合移入循环内部，实现实时更新 ---
                if doc_entities or doc_relations or successful_chunks:
                    doc_chunks_df = pd.DataFrame(successful_chunks)
                    fuse_and_update_knowledge_base(
                        doc_entities,
                        doc_relations,
                        doc_chunks_df,
                        rag_space_path,
                        embedding_config,
                        total_token_usage
                    )
                else:
                    print("ℹ️  No new knowledge to fuse for this document.")

            else:
                print("ℹ️  Skipping entity/relation extraction.")

    # <--- DELETED: 移除了循环外的批处理融合逻辑 ---

    # --- 最终的token消耗报告（仍然在所有文件处理完后打印） ---
    print(f"\n{'=' * 50}\n📊 Total Token Usage Report 📊")
    grand_total = 0
    for key, value in total_token_usage.items():
        print(f"   - {key.replace('_', ' ').title()}: {value:,} tokens")
        grand_total += value
    print(f"   ------------------------------------")
    print(f"   - Grand Total: {grand_total:,} tokens")
    print(f"{'=' * 50}")

    print(f"\n🏁 BiRAG processing pipeline finished. 🏁")


if __name__ == "__main__":
    main()
