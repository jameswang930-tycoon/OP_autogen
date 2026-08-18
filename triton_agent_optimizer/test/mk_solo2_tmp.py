# -*- coding: utf-8 -*-
import re
from pathlib import Path

base = Path(__file__).resolve().parent
src = (base / 'architecture_mermaid_single.html').read_text(encoding='utf-8')
head = re.search(r'<head>.*?</head>', src, re.S).group(0)
i4 = src.find('<h2>4.')
body_tail = src[i4:src.rfind('</body>')]
out = f"""<!DOCTYPE html>
<html lang="zh">
{head}
<body>
{body_tail}
</body>
</html>"""
p = base / 'arch4_solo.html'
p.write_text(out, encoding='utf-8')
print('solo 生成:', p.stat().st_size)