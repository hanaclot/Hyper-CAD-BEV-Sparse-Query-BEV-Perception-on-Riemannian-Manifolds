import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
code = open(r"E:\HyperCAD_BEV_Replication_2026\scripts\master_experiment.py", "r", encoding="utf-8").read()
lines = code.split("\n")
# Fix line 381-382 (0-indexed: 381)
for i, line in enumerate(lines):
    if "print(f'" in line and "Total elapsed" in line:
        # Fix the line to be proper
        lines[i] = "print(f'\\n  Total elapsed: {elapsed:.1f}s')"
# Also check for similar issues
code = "\n".join(lines)
open(r"E:\HyperCAD_BEV_Replication_2026\scripts\master_experiment.py", "w", encoding="utf-8").write(code)
print("Fixed successfully")
