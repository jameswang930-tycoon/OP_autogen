# -*- coding: utf-8 -*-
from pathlib import Path

p = Path(__file__).resolve().parent / 'architecture_mermaid_single.html'
t = p.read_text(encoding='utf-8')
D = 'class="mermaid"'
print('恢复后: 大小', len(t))
print('mermaid div:', t.count(D))
print('script 开/闭:', t.count('<script>'), '/', t.count('</script>'))
print('body 开/闭:', t.count('<body>'), '/', t.count('</body>'))
print('h2 4 存在:', '<h2>4.' in t)
print('initialize:', t.count('mermaid.initialize'))