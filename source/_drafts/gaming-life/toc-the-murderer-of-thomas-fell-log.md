```python3
# Python3 下的 bbcode 转 html

RULES = [
    ('<', '&lt;'),
    ('>', '&gt;'),

    (r'\[color=(.+?)\]', r'<span style="color: \1">'),
    (r'\[\/color\]', r'</span>'),

    (r'\[b\]', r'<strong>'),
    (r'\[\/b\]', r'</strong>'),
    (r'\[i\]', r'<i>'),
    (r'\[\/i\]', r'</i>'),
    (r'\[tt\]', r'<code>'),
    (r'\[\/tt\]', r'</code>'),

    (r'\n', r'<br />'),

    (r'<span style="color: silver">([^\|]+?)\|游戏外', r'<span style="color: mark-ingame">\1|游戏外'),
    
    (r'<span style="color: silver">', r'<span>'),
    
    (r'<span style="color: mark-ingame">', r'<span style="color: silver">'),
]

import re
def bbcode_to_html(text):
    for pat, repl in RULES:
        text = re.sub(pat, repl, text)
    return text
```
