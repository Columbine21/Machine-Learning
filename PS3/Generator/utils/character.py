"""Base characters dictionary & EOS (PAD) token"""

# Author: Ziqi Yuan <1564123490@qq.com>

from string import ascii_lowercase

EOS = '<EOS>'
idx_to_char = {i: c for i, c in enumerate(list(ascii_lowercase) + [EOS])}
char_to_idx = {c: i for i, c in enumerate(list(ascii_lowercase) + [EOS])}
EOF_IDX = char_to_idx[EOS]
PAD_IDX = len(idx_to_char)
