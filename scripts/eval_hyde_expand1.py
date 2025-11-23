#!/usr/bin/env python3
"""HyDE + 1 expansion"""
import sys
sys.path.insert(0, '/app')
exec(open('/app/scripts/eval_hyde_only.py').read().replace(
    'OUTPUT_PATH = "/app/results/eval_hyde_only_k100.json"',
    'OUTPUT_PATH = "/app/results/eval_hyde_expand1_k100.json"'
).replace(
    'NUM_EXPANDS = 0',
    'NUM_EXPANDS = 1'
).replace(
    '"HyDE only, no expansion"',
    '"HyDE + 1 expansion"'
).replace(
    'HyDE only',
    'HyDE + 1 expansion'
))
