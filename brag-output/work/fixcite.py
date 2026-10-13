import re
p = "brag-output/work/render.py"
s = open(p, encoding="utf-8").read()
pat = re.compile(r'            d\.text\(\(510, y\), .*?fill=GOLD\)')
new = '            d.text((510, y), "[2505.10468v4, p.6]  [p.3]", font=F36, fill=GOLD)'
s2, n = pat.subn(new, s, count=1)
assert n == 1, "pattern not found"
open(p, "w", encoding="utf-8").write(s2)
print("cleaned")
