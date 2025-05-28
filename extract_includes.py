#!/usr/bin/env python3
# extract_includes.py

import json, sys
from pathlib import Path

def main():
    # 默认读 build/compile_commands.json，可用第一个参数 override
    cc_path = Path(sys.argv[1] if len(sys.argv)>1 else "build/compile_commands.json")
    if not cc_path.exists():
        print(f"ERROR: {cc_path} 不存在，请先 cmake --build 或 cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON", file=sys.stderr)
        sys.exit(1)

    data = json.loads(cc_path.read_text())
    incs = set()

    for entry in data:
        # compile_commands.json 里可能是 "command" 字符串，也可能是 "arguments" 列表
        if "command" in entry:
            parts = entry["command"].split()
        else:
            parts = entry.get("arguments", [])

        i = 0
        while i < len(parts):
            tok = parts[i]
            if tok.startswith("-I"):
                # -I/path or -I /path 两种形式
                incs.add(tok[2:] if len(tok)>2 else parts[i+1])
                if tok == "-I": i += 1
            elif tok.startswith("-isystem"):
                incs.add(tok[8:] if len(tok)>8 else parts[i+1])
                if tok == "-isystem": i += 1
            i += 1

    for path in sorted(incs):
        print(path)

if __name__ == "__main__":
    main()
