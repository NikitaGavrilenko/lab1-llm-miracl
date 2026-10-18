"""MIRACL-ru: paired LLM reranking of annotated candidate passages."""
import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

MODELS = [
    'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.3',
    'google/gemma-2-9b-it', 'microsoft/Phi-3.5-mini-instruct',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'Qwen/Qwen3-8B',
]


def download(args):
    import requests
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = 'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-ru'
    for split in ('train', 'dev'):
        for kind in ('topics', 'qrels'):
            name = f'{kind}.miracl-v1.0-ru-{split}.tsv'
            target = out / name
            if target.exists():
                continue
            url = f'{root}/{kind}/{name}'
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            target.write_bytes(response.content)
            print('Downloaded', target)


def read_tsv(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.reader(f, delimiter='\t'))


def prepare(args):
    folder = Path(args.data)
    topics = {row[0]: row[1] for row in read_tsv(folder / 'topics.miracl-v1.0-ru-dev.tsv') if len(row) >= 2}
    qrels = {}
    for row in read_tsv(folder / 'qrels.miracl-v1.0-ru-dev.tsv'):
        if len(row) >= 4:
            qrels.setdefault(row[0], {})[row[2]] = int(row[3])
    # Official MIRACL contains an independent corpus; this experiment uses only
    # judged positives and negatives, NOT full-corpus retrieval.
    from datasets import load_dataset
    dataset = load_dataset('miracl/miracl', 'ru', split='dev', trust_remote_code=True)
    rng = random.Random(args.seed)
    records = []
    missing = 0
    for row in dataset:
        qid = str(row['query_id'])
        if qid not in topics or qid not in qrels:
            missing += 1
            continue
        docs = {}
        for label, key in ((1, 'positive_passages'), (0, 'negative_passages')):
            for d in row[key]:
                docid = str(d['docid'])
                docs[docid] = {'id': docid, 'title': d['title'], 'text': d['text'], 'rel': qrels[qid].get(docid, label)}
        docs = list(docs.values())
        if not docs or not any(x['rel'] > 0 for x in docs) or not any(x['rel'] == 0 for x in docs):
            continue
        docs.sort(key=lambda x: x['id'])
        records.append({'qid': qid, 'query': topics[qid], 'docs': docs})
    rng.shuffle(records)
    records = records[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    stats = {'dataset': 'MIRACL ru dev', 'sampling_seed': args.seed, 'queries': len(records),
             'documents_per_query': dict(Counter(len(r['docs']) for r in records)),
             'missing_query_or_qrels': missing,
             'notes': 'Only judged candidates; not full-corpus retrieval or agentic multi-hop evaluation.'}
    out.with_suffix('.passport.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def tokenize(text):
    return re.findall(r'\w+', text.casefold())


def bm25_rank(query, docs):
    from rank_bm25 import BM25Okapi
    corpus = [tokenize(d['title'] + ' ' + d['text']) or ['_'] for d in docs]
    scores = BM25Okapi(corpus).get_scores(tokenize(query))
    return [docs[int(i)]['id'] for i in np.argsort(-scores, kind='stable')]


def prompt(query, docs, shots):
    intro = ('Отранжируй фрагменты по релевантности запросу. Используй только предоставленные ID. '
             'Ответь строго JSON: {"ranking": ["id1", "id2", ...]}. '
             'Каждый ID должен встречаться ровно один раз. Не добавляй текст вне JSON.')
    examples = ''
    if shots:
        examples = ('\nПример 1. Запрос: столица Франции. A: Париж — столица Франции. '
                    'B: Берлин — столица Германии. Ответ: {"ranking":["A","B"]}.\n'
                    'Пример 2. Запрос: спутник Земли. A: Марс — планета. '
                    'B: Луна — естественный спутник Земли. Ответ: {"ranking":["B","A"]}.\n')
    shown = '\n'.join(f'{i+1}. ID={d["id"]}; {d["title"]}; {d["text"][:650]}' for i, d in enumerate(docs))
    return intro + examples + '\nЗапрос: ' + query + '\nКандидаты:\n' + shown


def parse_ranking(s, docids):
    try:
        obj = json.loads(s)
        ids = obj['ranking']
        if isinstance(ids, list) and len(ids) == len(docids) and all(isinstance(i, str) for i in ids) and set(ids) == set(docids):
            return ids
    except (ValueError, KeyError, TypeError):
        pass
    return None


def score(ranking, rel, k=10):
    def dcg(order):
        return sum((2 ** rel.get(doc, 0) - 1) / np.log2(i + 2) for i, doc in enumerate(order[:k]))
    ideal = sorted(rel, key=lambda x: rel[x], reverse=True)
    denom = dcg(ideal)
    ndcg = float(dcg(ranking) / denom) if denom else 0.0
    total = sum(v > 0 for v in rel.values())
    recall = sum(rel.get(doc, 0) > 0 for doc in ranking[:k]) / total if total else 0.0
    reciprocal = next((1.0 / (i + 1) for i, doc in enumerate(ranking) if rel.get(doc, 0) > 0), 0.0)
    return round(ndcg, 5), round(recall, 5), round(reciprocal, 5)


def evaluate(args):
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url or os.environ.get('OPENAI_BASE_URL'),
                    api_key=os.environ.get('OPENAI_API_KEY', 'not-needed'))
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding='utf-8').splitlines() if line.strip()]
    models = args.models or MODELS
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if out.exists():
        with out.open(encoding='utf-8') as f:
            completed = {(r['model'], r['mode'], r['qid']) for r in csv.DictReader(f)}
    columns = ['model', 'mode', 'qid', 'ndcg10', 'recall10', 'mrr', 'invalid', 'latency_s', 'error', 'ranking']
    with out.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, columns)
        if f.tell() == 0:
            writer.writeheader()
        for model in models:
            for mode in ('zero-shot', 'few-shot'):
                for row in rows:
                    if (model, mode, row['qid']) in completed:
                        continue
                    docs = row['docs']
                    gold = {d['id']: d['rel'] for d in docs}
                    baseline = bm25_rank(row['query'], docs)
                    start = time.perf_counter()
                    error = ''
                    try:
                        response = client.chat.completions.create(
                            model=model, temperature=0, max_tokens=args.max_tokens,
                            messages=[{'role': 'user', 'content': prompt(row['query'], docs, mode == 'few-shot')}],
                            timeout=args.timeout)
                        raw = response.choices[0].message.content or ''
                        ranking = parse_ranking(raw, set(gold))
                        invalid = int(ranking is None)
                        if invalid:
                            ranking = baseline  # explicitly tracked fallback
                    except Exception as e:
                        ranking = baseline
                        invalid = 1
                        error = str(e)[:150]
                    ndcg, recall, mrr = score(ranking, gold)
                    writer.writerow({'model': model, 'mode': mode, 'qid': row['qid'],
                                     'ndcg10': ndcg, 'recall10': recall, 'mrr': mrr,
                                     'invalid': invalid, 'latency_s': round(time.perf_counter()-start, 3),
                                     'error': error, 'ranking': json.dumps(ranking, ensure_ascii=False)})
                    f.flush()
        # Baseline stored only once per query, independently of LLM.


def summarize(args):
    import pandas as pd
    df = pd.read_csv(args.input)
    rng = np.random.default_rng(42)
    output = []
    for (model, mode), group in df.groupby(['model', 'mode']):
        item = {'model': model, 'mode': mode, 'queries': len(group)}
        for metric in ('ndcg10', 'recall10', 'mrr'):
            a = group[metric].to_numpy(dtype=float)
            if len(a) > 1:
                boots = np.mean(rng.choice(a, (2000, len(a)), replace=True), axis=1)
                low, high = np.quantile(boots, [.025, .975])
            else:
                low = high = float(np.mean(a))
            item[metric] = round(float(np.mean(a)), 4)
            item[metric+'_95ci'] = f'{low:.4f}–{high:.4f}'
        item['invalid_rate'] = round(float(group['invalid'].mean()), 4)
        item['latency_p95_s'] = round(float(group['latency_s'].quantile(.95)), 3)
        output.append(item)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).sort_values('ndcg10', ascending=False).to_csv(path, index=False)
    print(pd.DataFrame(output).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('download'); d.add_argument('--out', default='data/raw'); d.set_defaults(fn=download)
    p = sub.add_parser('prepare'); p.add_argument('--data', default='data/raw'); p.add_argument('--out', default='data/ru_dev_sample.jsonl'); p.add_argument('--limit', type=int, default=100); p.add_argument('--seed', type=int, default=42); p.set_defaults(fn=prepare)
    e = sub.add_parser('evaluate'); e.add_argument('--input', default='data/ru_dev_sample.jsonl'); e.add_argument('--out', default='results/raw.csv'); e.add_argument('--models', nargs='+'); e.add_argument('--base-url'); e.add_argument('--timeout', type=int, default=90); e.add_argument('--max-tokens', type=int, default=1024); e.set_defaults(fn=evaluate)
    s = sub.add_parser('summarize'); s.add_argument('--input', default='results/raw.csv'); s.add_argument('--out', default='results/summary.csv'); s.set_defaults(fn=summarize)
    args = parser.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
