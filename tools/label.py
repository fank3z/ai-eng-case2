import json
p = 'evals/human_labels.jsonl'
rows = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
def save():
    open(p, 'w', encoding='utf-8').write('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
for r in rows:
    if r.get('human_verdict'):
        continue
    print('\n' + '=' * 70)
    print(r['id'], 'ПИТАННЯ:', r['question'])
    print('ЕТАЛОН:   ', r['expected_answer'])
    print('ВІДПОВІДЬ:', r['answer'])
    k = input('p = pass, f = fail, s = пропустити, q = вийти: ').strip().lower()
    if k == 'q':
        break
    if k in ('p', 'f'):
        r['human_verdict'] = 'pass' if k == 'p' else 'fail'
        n = input('Примітка, чому так (Enter — без): ').strip()
        if n:
            r['note'] = n
        save()
print('\nРозмічено:', sum(1 for r in rows if r.get('human_verdict')))
