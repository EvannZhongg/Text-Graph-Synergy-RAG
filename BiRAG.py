# BiRAG.py

import sys
from pathlib import Path
import hashlib
import json
import yaml
import pandas as pd
import asyncio

# 从我们的模块中导入所需函数
from pdf2md import process_pdf
from chunks import chunk_dispatcher
from embedding import generate_chunk_embeddings
from extraction import extract_entities_and_relations
from fusion import fuse_and_update_knowledge_base


def main():
    """
    主调度函数，自动化处理 input 文件夹内的所有文档，并实时增量更新全局知识库。
    """
    print("🚀 Starting BiRAG processing pipeline...")

    input_dir = Path("input")
    output_dir_base = Path("rag_storage")
    config_path = Path("config.yaml")

    # --- 加载所有配置 ---
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

    rag_space_path = Path(rag_space)  # 全局知识库的顶级目录
    input_dir.mkdir(exist_ok=True)
    output_dir_base.mkdir(exist_ok=True)

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

            # <--- 新增核心逻辑：在此处为每个 chunk 添加源文件名 ---
            source_filename = doc_path.name
            for chunk in chunks:
                chunk['source_document_name'] = source_filename
            # ----------------------------------------------------

            chunks_dir = unique_output_dir / "chunks"
            chunks_dir.mkdir(exist_ok=True)

            with open(chunks_dir / "chunks.json", 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)

            successful_chunks = []
            if embedding_config.get('api_key'):
                # 2. 生成向量
                chunks_with_embeddings = generate_chunk_embeddings(chunks, embedding_config)
                successful_chunks = [c for c in chunks_with_embeddings if c.get('embedding') is not None]

                if successful_chunks:
                    df_chunks = pd.DataFrame(successful_chunks)
                    # 调整列顺序，把新列放在前面
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
                doc_entities, doc_relations = asyncio.run(extract_entities_and_relations(
                    successful_chunks, llm_config, extraction_config
                ))

                kg_dir = unique_output_dir / "kg"
                kg_dir.mkdir(exist_ok=True)
                with open(kg_dir / "entities.json", 'w', encoding='utf-8') as f:
                    json.dump(doc_entities, f, indent=2, ensure_ascii=False)
                with open(kg_dir / "relations.json", 'w', encoding='utf-8') as f:
                    json.dump(doc_relations, f, indent=2, ensure_ascii=False)
                print(f"💾 Raw extraction results saved to: {kg_dir}")

                # 4. 立即进行全局知识融合
                if doc_entities or doc_relations or successful_chunks:
                    doc_chunks_df = pd.DataFrame(successful_chunks)
                    fuse_and_update_knowledge_base(
                        doc_entities,
                        doc_relations,
                        doc_chunks_df,
                        rag_space_path,
                        embedding_config
                    )
                else:
                    print("ℹ️  No new knowledge to fuse for this document.")

            else:
                print("ℹ️  Skipping entity/relation extraction.")

    print(f"\n{'=' * 50}\n🏁 BiRAG processing pipeline finished. 🏁")


if __name__ == "__main__":
    main()