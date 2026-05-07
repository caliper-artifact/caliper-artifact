#!/usr/bin/env python3
import json
from collections import OrderedDict
import sys


def sort_json_keys_numeric(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sorted_data = OrderedDict(
        (k, data[k]) for k in sorted(data.keys(), key=lambda x: int(x))
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_data, f, indent=2, ensure_ascii=False)
        f.write('\n')  # newline at end of file


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <input.json> <output.json>')
        sys.exit(1)
    sort_json_keys_numeric(sys.argv[1], sys.argv[2])
