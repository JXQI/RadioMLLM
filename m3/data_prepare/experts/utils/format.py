from typing import Dict, List


def _is_valid_conversation(conversation: Dict) -> bool:
    """
    检查单个对话记录是否有效

    Args:
        conversation: 对话记录字典

    Returns:
        bool: 如果对话有效返回True，否则返回False
    """
    # 检查基础结构
    if not isinstance(conversation, Dict) or "conversations" not in conversation:
        return False

    # 检查对话列表是否为空
    if len(conversation["conversations"]) == 0:
        return False

    from_gpt = 0
    from_human = 0

    # 检查每条对话的完整性
    for _conv in conversation["conversations"]:
        # 检查必需字段
        if ("from" not in _conv) or ("value" not in _conv):
            return False

        # 检查GPT对话的特殊字段
        if _conv["from"] == "gpt":
            if (
                ("thoughts" not in _conv)
                or ("actions" not in _conv)
                or ("value" not in _conv)
            ):
                return False

        # 统计对话来源
        if _conv["from"] == "gpt":
            from_gpt += 1
        elif _conv["from"] == "human":
            from_human += 1

    # 检查人类和GPT对话数量是否匹配
    if from_human != from_gpt:
        print(f"对话次数不同: GPT={from_gpt}, Human={from_human}")
        return False

    return True


def filter_valid_conversations(conversations: List[Dict]) -> List[Dict]:
    """
    过滤出有效的对话记录

    Args:
        conversations: 原始对话记录列表

    Returns:
        List[Dict]: 过滤后的有效对话记录列表
    """
    valid_conversations = []

    for conversation in conversations:
        if _is_valid_conversation(conversation):
            valid_conversations.append(conversation)

    return valid_conversations
