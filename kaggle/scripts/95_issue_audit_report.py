#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/kaggle/work_here/yolov13')
raw_path = ROOT / 'kaggle/reports/upstream_issues_raw.json'
cat_path = ROOT / 'kaggle/reports/upstream_issues_categorized.json'
out_path = ROOT / 'kaggle/reports/UPSTREAM_ISSUES_AUDIT.md'

issues = json.loads(raw_path.read_text(encoding='utf-8'))
cats = {x['number']: x for x in json.loads(cat_path.read_text(encoding='utf-8'))}

focus_numbers = {15, 21, 37, 40, 48, 53, 54, 61, 65, 72, 74, 75}

fix_notes = {
    15: 'DDP subprocess import path hardened in `ultralytics/utils/dist.py` to avoid custom module resolution failures.',
    21: 'Same root cause as #15; local source import precedence enforced for DDP temp runner.',
    37: 'Added non-finite loss guard in train loop and safer DDP settings in `ultralytics/engine/trainer.py`.',
    40: 'Mitigated DDP grad-stride instability by setting `gradient_as_bucket_view=False` in DDP wrapper.',
    48: 'Applied same DDP stability fixes and validated with distributed runs on 2x T4.',
    53: 'Applied same DDP stability fixes and validated with distributed runs on 2x T4.',
    54: 'Added TensorRT export script and test path in `kaggle/scripts/70_export_onnx_tensorrt.sh`.',
    61: 'Added ONNX export testing path and parameter matrix in scripts for reproducibility.',
    65: 'Documented augmentation edge-case handling in dependency and smoke workflows.',
    72: 'Not fully resolved: upstream feature parity limitations (segment variants) remain architecture-specific.',
    74: 'Not fully resolved: OBB/feature availability and model support limitations remain upstream dependent.',
    75: 'Partially addressed through robust run scripts; dataset-key mismatches still require dataset yaml correction.',
}

lines = []
lines.append('# Upstream Issues Audit (iMoonLab/yolov13)')
lines.append('')
lines.append('This audit summarizes upstream issues and how they were handled in this fork.')
lines.append('')
lines.append('## Scope Summary')
lines.append('')
lines.append(f'- Total upstream issues reviewed: {len(issues)}')
lines.append(f'- Focused issues in this pass: {len(focus_numbers)}')
lines.append('- Priority focus: DDP stability, export reproducibility, Kaggle workflow reliability')
lines.append('')
lines.append('## Resolution Matrix')
lines.append('')
lines.append('| Issue | State | Category | Status in this fork | Notes |')
lines.append('|---|---|---|---|---|')

for i in sorted(issues, key=lambda x: x['number']):
    num = i['number']
    if num not in focus_numbers:
        continue
    category = ','.join(cats.get(num, {}).get('categories', ['other']))
    note = fix_notes.get(num, 'Tracked only')
    status = 'resolved' if num in {15, 21, 40, 48, 53, 54, 61} else ('partial' if num in {37, 65, 75} else 'not_resolved')
    lines.append(f"| [#{num}]({i['html_url']}) | {i['state']} | {category} | {status} | {note} |")

lines.append('')
lines.append('## Notes')
lines.append('')
lines.append('- Some upstream issues are architectural feature requests (e.g., full OBB/segment/pose parity) and cannot be fully solved by runtime hardening alone.')
lines.append('- This fork prioritizes training pipeline reliability, distributed compatibility, and export workflow reproducibility on Kaggle.')

out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'Wrote {out_path}')
