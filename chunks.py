import tiktoken
import re
from typing import List, Dict

# 加载编码器
ENCODING = tiktoken.get_encoding("cl100k_base")


def _get_token_count(text: str) -> int:
    """辅助函数：计算文本的 token 数量"""
    return len(ENCODING.encode(text))


def _preprocess_text(text: str) -> str:
    """
    文本预处理器：在分块前清洗掉 PDF 转 Markdown 插入的图片占位符和 Markdown 图片链接。
    """
    # 1. 移除 PDF 转 Markdown 生成的图片占位符，例如 <!-- image -->
    cleaned_text = re.sub(r'<!--\s*image\s*-->', '', text)

    # 2. 移除所有 Markdown 图片链接，例如 ![alt](url)
    cleaned_text = re.sub(r'\n?!\[.*?\]\(.*?\)\n?', '', cleaned_text)

    return cleaned_text.strip()


def chunk_text_fixed_size(
        text: str,
        file_hash: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150
) -> List[Dict]:
    """固定大小的分块策略 (此函数不变)"""
    if not text:
        return []
    tokens = ENCODING.encode(text)
    chunks = []
    step = chunk_size - chunk_overlap
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i: i + chunk_size]
        chunk_text = ENCODING.decode(chunk_tokens)
        chunk_id = f"{file_hash}_{len(chunks)}"
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "token_count": len(chunk_tokens)
        })
    print(f"✅ 'fixed' 策略分块完成，共计 {len(chunks)} 个块。")
    return chunks


def chunk_text_semantic(
        text: str,
        file_hash: str,
        target_size: int = 1000,
        overlap_target: int = 150
) -> List[Dict]:
    """高级语义分块策略 V2 (现在处理的是已清洗过的文本)"""
    if not text:
        return []

    # 图片链接已在预处理中被移除，现在我们只按标题和段落切分
    # 我们以 ## 或更高级别的标题作为主要分隔符
    separators_regex = r'(\n##+\s.*)'
    blocks = re.split(separators_regex, text)

    structured_blocks = []
    i = 0
    while i < len(blocks):
        block = blocks[i].strip()
        if not block:
            i += 1
            continue
        if re.match(separators_regex, block) and i + 1 < len(blocks):
            next_block = blocks[i + 1].strip()
            structured_blocks.append(f"{block}\n\n{next_block}")
            i += 2
        else:
            structured_blocks.append(block)
            i += 1

    if not structured_blocks:
        return [
            {
                "chunk_id": f"{file_hash}_0",
                "text": text,
                "token_count": _get_token_count(text)
            }
        ] if text else []

    # 后续的贪心算法组合逻辑保持不变
    base_chunks_of_blocks = []
    # ... (此部分代码与上一版完全相同，此处省略以保持简洁)
    current_base_chunk = []
    current_tokens = 0
    for block in structured_blocks:
        block_tokens = _get_token_count(block)
        if current_tokens + block_tokens > target_size and current_base_chunk:
            base_chunks_of_blocks.append(current_base_chunk)
            current_base_chunk = [block]
            current_tokens = block_tokens
        else:
            current_base_chunk.append(block)
            current_tokens += block_tokens
    if current_base_chunk:
        base_chunks_of_blocks.append(current_base_chunk)

    final_chunks = []
    for i, base_chunk_blocks in enumerate(base_chunks_of_blocks):
        final_chunk_content_blocks = list(base_chunk_blocks)
        if i + 1 < len(base_chunks_of_blocks):
            next_base_chunk_blocks = base_chunks_of_blocks[i + 1]
            overlap_tokens = 0
            for block_to_add in next_base_chunk_blocks:
                block_to_add_tokens = _get_token_count(block_to_add)
                if block_to_add_tokens > overlap_target or (overlap_tokens + block_to_add_tokens) > overlap_target:
                    final_chunk_content_blocks.append(block_to_add)
                    break
                final_chunk_content_blocks.append(block_to_add)
                overlap_tokens += block_to_add_tokens
        final_text = "\n\n".join(final_chunk_content_blocks)
        chunk_id = f"{file_hash}_{len(final_chunks)}"
        final_chunks.append({
            "chunk_id": chunk_id,
            "text": final_text,
            "token_count": _get_token_count(final_text)
        })

    print(f"✅ 'semantic' 策略分块完成，共计 {len(final_chunks)} 个块。")
    return final_chunks


def chunk_dispatcher(text: str, file_hash: str, config: Dict) -> List[Dict]:
    """
    分块调度器：在分发任务前，先进行统一的文本预处理。
    """
    # --- 核心改动：在这里统一调用预处理器 ---
    print("🧹 Preprocessing text to remove image placeholders and links...")
    cleaned_text = _preprocess_text(text)

    strategy = config.get('strategy', 'fixed')

    if strategy == 'semantic':
        return chunk_text_semantic(
            cleaned_text,
            file_hash,
            target_size=config.get('semantic_target', 1000),
            overlap_target=config.get('semantic_overlap', 150)
        )
    elif strategy == 'fixed':
        return chunk_text_fixed_size(
            cleaned_text,
            file_hash,
            chunk_size=config.get('fixed_size', 1000),
            chunk_overlap=config.get('fixed_overlap', 150)
        )
    else:
        print(f"⚠️  Warning: Unknown chunking strategy '{strategy}'. Falling back to 'fixed'.")
        return chunk_text_fixed_size(cleaned_text, file_hash)