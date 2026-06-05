"""
Prompt 加载工具
从 prompts/ 目录下的 YAML 文件加载和管理提示词模板
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class PromptLoader:
    """从 YAML 文件加载 prompt 模板"""

    def __init__(self, prompts_dir: Optional[str] = None):
        if prompts_dir is None:
            prompts_dir = os.path.join(Path(__file__).parent.parent.parent, "prompts")
        self.prompts_dir = prompts_dir

    def _load_yaml(self, relative_path: str) -> Dict[str, Any]:
        """加载单个 YAML prompt 文件"""
        full_path = os.path.join(self.prompts_dir, relative_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Prompt file not found: {full_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_system_prompt(self, relative_path: str) -> str:
        """加载 system prompt - 返回纯文本内容"""
        data = self._load_yaml(relative_path)
        messages = data.get("messages", [])
        for msg in messages:
            if msg.get("role") == "system":
                return msg["content"].strip()
        raise ValueError(f"No system message found in {relative_path}")

    def load_user_prompt(self, relative_path: str) -> str:
        """加载 user prompt - 返回纯文本内容"""
        data = self._load_yaml(relative_path)
        messages = data.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                return msg["content"].strip()
        raise ValueError(f"No user message found in {relative_path}")

    def load_messages(self, relative_path: str) -> list:
        """加载所有消息列表 [("role", "content"), ...]"""
        data = self._load_yaml(relative_path)
        messages = data.get("messages", [])
        return [(msg["role"], msg["content"]) for msg in messages]

    def load_system_with_format(self, relative_path: str) -> str:
        """加载 system prompt (保留 {variable} 占位符)"""
        return self.load_system_prompt(relative_path)

    def load_template(self, relative_path: str) -> str:
        """加载模板内容（用于 wrapper 类型的 prompt）"""
        data = self._load_yaml(relative_path)
        template = data.get("template", "")
        if not template:
            raise ValueError(f"No template found in {relative_path}")
        return template.strip()


# 全局单例
_default_loader: Optional[PromptLoader] = None


def get_prompt_loader(prompts_dir: Optional[str] = None) -> PromptLoader:
    """获取 prompt loader 单例"""
    global _default_loader
    if _default_loader is None or prompts_dir is not None:
        _default_loader = PromptLoader(prompts_dir)
    return _default_loader
