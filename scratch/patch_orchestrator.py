import re

path = 'backend/services/orchestrator.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace _should_run_critic body
should_run_pattern = r'(def _should_run_critic.*?)(?=\ndef _route_by_critique)'
def should_run_repl(m):
    header = m.group(1).split('"""')[0] + '"""' + m.group(1).split('"""')[1] + '"""\n'
    return header + '    # Disabled: Groq free-tier 429 fix (re-enable after upgrade)\n    return "synthesize_response"\n'

content = re.sub(should_run_pattern, should_run_repl, content, flags=re.DOTALL)

# Replace _route_by_critique body
route_by_pattern = r'(def _route_by_critique.*?)(?=\n# ════════════════════════════════════════════════════════════════\n# Graph builder)'
def route_by_repl(m):
    header = m.group(1).split('"""')[0] + '"""' + m.group(1).split('"""')[1] + '"""\n'
    return header + '    # Disabled: Groq free-tier 429 fix (re-enable after upgrade)\n    return "synthesize_response"\n'

content = re.sub(route_by_pattern, route_by_repl, content, flags=re.DOTALL)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched orchestrator.py successfully.")
