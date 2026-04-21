import os

path = r"backend/api/app.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if "allow_origins=[" in line or "allow_origins=get_settings().cors_origins" in line:
        new_lines.append("    allow_origins=[\n")
        new_lines.append("        'http://localhost:3000',\n")
        new_lines.append("        'http://localhost:3001',\n")
        new_lines.append("        'http://localhost:5173',\n")
        new_lines.append("        'https://app.aifinbrain.io',\n")
        new_lines.append("    ],\n")
        skip = True
        continue
    if skip:
        if "]," in line or ")," in line or "cors_origins," in line:
            skip = False
        continue
    new_lines.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("Updated app.py with hardcoded port 3001 for CORS")
