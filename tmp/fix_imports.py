import os

path = r"backend/services/agent_factory.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if '("agents.' in line:
        new_lines.append(line.replace('("agents.', '("backend.agents.'))
    else:
        new_lines.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("Updated agent_factory.py")
