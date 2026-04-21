import os

path = r"backend/api/app.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if "allow_origins=[" in line:
        new_lines.append("    allow_origins=get_settings().cors_origins,\n")
        skip = True
        continue
    if skip:
        if "]" in line:
            skip = False
        continue
    new_lines.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("Updated app.py with CORS settings")
