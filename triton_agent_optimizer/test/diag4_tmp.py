# -*- coding: utf-8 -*-
import re
from pathlib import Path

t = Path(__file__).resolve().parent.joinpath('architecture_mermaid_single.html').read_text(encoding='utf-8')
print('文件大小:', len(t))
D = '<div class="mermaid">'
print('mermaid div 数:', t.count(D))
print('<div 总数:', len(re.findall(r'<div', t)))
print('</div> 总数:', len(re.findall(r'</div>', t)))
print('<script> 数:', t.count('<script>'))
print('</script> 数:', t.count('</script>'))
print('<head>:', t.count('<head>'), ' </head>:', t.count('</head>'))
print('<body>:', t.count('<body>'), ' </body>:', t.count('</body>'))
print('mermaid.initialize:', t.count('mermaid.initialize'))
for m in re.finditer(r'<h2', t):
    print('h2 at', m.start(), t[m.start():m.start() + 40].split('\n')[0])
# 检查 head 之后是否有 mermaid script 加载顺序
i_script = t.find('<script>')
print('第一个 <script> at', i_script, '之前内容:', repr(t[max(0, i_script - 80):i_script])[-80:])