from typing import Dict, Any

PROMPTS: Dict[str, Any] = {}

# --- 核心分隔符定义 ---
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# --- 系统级指令，定义了 LLM 的角色和所有提取规则 ---
PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    * **Identification:** Identify clearly defined and meaningful entities in the input text.
    * **Entity Details:** For each identified entity, extract:
        * `entity_name`: The name of the entity. Use title case for consistency.
        * `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none apply, classify as `other`.
        * `entity_description`: A concise description based *solely* on the input text.
    * **Output Format - Entities:** Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    * **Identification:** Identify direct and meaningful relationships between previously extracted entities. Decompose N-ary relationships into binary pairs.
    * **Relationship Details:** For each binary relationship, extract:
        * `source_entity`: The source entity name, ensuring consistency.
        * `target_entity`: The target entity name, ensuring consistency.
        * `relationship_keywords`: Comma-separated high-level keywords summarizing the relationship.
        * `relationship_description`: A concise explanation of the connection.
    * **Output Format - Relationships:** Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **General Rules:**
    * Output all entities first, then all relationships.
    * All output must be in the third person, avoiding pronouns like 'I', 'you', 'this article'.
    * The entire output must be in {language}. Proper nouns should be retained in their original language.
    * Signal the end of all extractions by outputting the literal string `{completion_delimiter}` on the final line.

---Real Data to be Processed---
<Input>
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
"""

# --- 首次提取的用户指令 ---
PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract entities and relationships from the input text provided in the system prompt.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list. Do not include any introductory or concluding remarks.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line.
4.  **Output Language:** Ensure the output is in {language}.

<Output>
"""

# --- 二次修正（Gleaning）的用户指令 ---
PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Focus on Corrections/Additions:**
    * **Do NOT** re-output items that were **correctly and fully** extracted in the last task.
    * If an item was **missed**, extract and output it now.
    * If an item was **truncated, had missing fields, or was otherwise incorrectly formatted**, re-output the *corrected and complete* version.
2.  **Strict Adherence to Format:** All output must follow the format specified in the system prompt.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line.
4.  **Output Language:** Ensure the output is in {language}.

<Output>
"""