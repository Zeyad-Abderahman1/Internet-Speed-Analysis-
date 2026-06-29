import os
import json

files = {
    'Internet_Speed_Project_Enhanced.ipynb': None,
    'dashboard.html': 100_000,
    'best_model.pkl': None,
    'pairplot.png': None,
    'README.md': None,
    'requirements.txt': None,
}

print('File verification:')
all_ok = True
for fname, min_size in files.items():
    exists = os.path.exists(fname)
    size = os.path.getsize(fname) if exists else 0
    size_ok = (min_size is None or size >= min_size)
    status = 'PASS' if (exists and size_ok) else 'FAIL'
    if not (exists and size_ok):
        all_ok = False
    print(f'  [{status}]  {fname}  ({size/1024:.1f} KB)')

print()

# Check notebook for errors
with open('Internet_Speed_Project_Enhanced.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

error_cells = [
    c for c in nb['cells']
    if c['cell_type'] == 'code'
    and any(o.get('output_type') == 'error' for o in c.get('outputs', []))
]

if error_cells:
    print(f'[FAIL]  {len(error_cells)} cell(s) with errors:')
    for c in error_cells:
        errs = [o for o in c['outputs'] if o.get('output_type') == 'error']
        for e in errs:
            ename = e['ename']
            evalue = e['evalue']
            print(f'   {ename}: {evalue}')
else:
    print('[PASS]  No error cells in notebook')

total_cells = len(nb['cells'])
executed    = sum(1 for c in nb['cells']
                  if c['cell_type'] == 'code' and c.get('execution_count'))

print(f'[INFO]  Total cells    : {total_cells}')
print(f'[INFO]  Executed cells : {executed}')
print()
print('Overall:', 'ALL PASS' if (all_ok and not error_cells) else 'SOME ISSUES — see above')
