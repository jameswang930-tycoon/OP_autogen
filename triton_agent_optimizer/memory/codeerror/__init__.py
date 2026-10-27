"""代码错误记忆系统 — 记录、检索、避免重复错误。"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

_STORE = Path(__file__).resolve().parent


class CodeErrorMemory:
    """每个 kernel 一个 JSON 文件，存储报错 → 解决方案的映射。

    结构:
    {
      "kernel": "softmax",
      "errors": [
        {
          "pattern": "not found in kernel.py",
          "count": 3,
          "solved": true,
          "solution": "保持原函数名不变，不要重命名。在 @triton.jit 下用 def softmax_kernel(...)",
          "last_seen": "2026-07-29T10:00:00"
        },
        {
          "pattern": "operands could not be broadcast",
          "count": 5,
          "solved": false,
          "solution": null,
          "last_seen": "2026-07-29T10:05:00"
        }
      ]
    }
    """

    def __init__(self, kernel_name: str):
        self.kernel_name = kernel_name
        self.path = _STORE / f"{kernel_name}.json"
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"kernel": self.kernel_name, "errors": []}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    def find_solution(self, error_msg: str) -> Optional[str]:
        """查找已知的解决方案。匹配 error pattern。"""
        for err in self.data["errors"]:
            if err["pattern"] in error_msg:
                if err.get("solved") and err.get("solution"):
                    return err["solution"]
        return None

    def record_error(self, error_msg: str):
        """记录一次错误发生（不计入重复）。提取关键 pattern。"""
        pattern = self._extract_pattern(error_msg)
        for err in self.data["errors"]:
            if err["pattern"] == pattern:
                err["count"] += 1
                err["last_seen"] = datetime.now().isoformat()
                self._save()
                return
        # 新错误
        self.data["errors"].append({
            "pattern": pattern,
            "count": 1,
            "solved": False,
            "solution": None,
            "last_seen": datetime.now().isoformat(),
        })
        self._save()

    def record_solution(self, error_msg: str, solution: str):
        """记录一次成功解决。"""
        pattern = self._extract_pattern(error_msg)
        for err in self.data["errors"]:
            if err["pattern"] == pattern:
                err["solved"] = True
                err["solution"] = solution
                err["last_seen"] = datetime.now().isoformat()
                self._save()
                return
        self.data["errors"].append({
            "pattern": pattern,
            "count": 1,
            "solved": True,
            "solution": solution,
            "last_seen": datetime.now().isoformat(),
        })
        self._save()

    def get_consecutive_count(self, error_msg: str) -> int:
        """检查这个错误最近已连续出现多少次（用于判断是否卡住）。"""
        pattern = self._extract_pattern(error_msg)
        for err in self.data["errors"]:
            if err["pattern"] == pattern:
                return err.get("count", 0)
        return 0

    def get_all_solutions(self) -> str:
        """返回所有已解决的方案文本（注入 Coder prompt）。"""
        solved = [e for e in self.data["errors"] if e.get("solved") and e.get("solution")]
        if not solved:
            return ""
        lines = ["## 已知错误及修复方案（来自之前轮次）"]
        for e in solved:
            lines.append(f"- 错误: {e['pattern'][:80]}")
            lines.append(f"  修复: {e['solution'][:200]}")
        return "\n".join(lines)

    @staticmethod
    def _extract_pattern(error_msg: str) -> str:
        """提取纯英文关键词 pattern（去掉数字、变量名、shape 值）。"""
        import re
        # 只保留英文单词 + 常见 Triton/python 关键词
        words = re.findall(r"[a-zA-Z_]+", error_msg[:300])
        # 过滤单词 (去掉太短的、纯数字的)
        keywords = [w for w in words if len(w) > 3
                    and w not in ("this", "that", "with", "from", "been", "were",
                                  "when", "will", "have", "they", "them", "then",
                                  "than", "also", "into", "just", "like", "make",
                                  "more", "over", "some", "such", "take", "very",
                                  "your", "here", "each")]
        # 找关键 pattern 短语 (去重取前 10 个)
        seen = set()
        result = []
        for kw in keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                result.append(kw)
        return " ".join(result[:10])

    def search_all(self, error_msg: str) -> str:
        """跨所有 kernel 搜索匹配的已知错误。"""
        pattern = self._extract_pattern(error_msg)
        pattern_words = set(pattern.lower().split())
        results = []
        for f in sorted(_STORE.glob("*.json")):
            if f.name == self.path.name:
                continue  # skip current kernel (already searched)
            data = json.loads(f.read_text(encoding="utf-8"))
            for err in data.get("errors", []):
                err_words = set(err["pattern"].lower().split())
                # 至少 3 个单词匹配
                overlap = pattern_words & err_words
                if len(overlap) >= 3 and err.get("solved") and err.get("solution"):
                    results.append(f"[{data['kernel']}] {err['solution'][:200]}")
        return "\n".join(results[:3]) if results else ""
