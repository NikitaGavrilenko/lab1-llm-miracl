"""Prepare a reproducible, leakage-screened MIRACL-ru judged-pool experiment."""
import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import requests

OFFICIAL_REVISION = '5be20db9509754dadad47689368639fcec739c00'
MIRROR_REVISION = '1fd7b335ab9be859040f664efb07b9665d4bf8ee'
SEED = 42
SIZES = {'train': 64, 'validation': 16, 'test': 40, 'application': 20}
THRESHOLD = 0.85
OFFICIAL_CORPUS_REVISION = 'd921ec7e349ce0d28daf30b2da9da5ee698bef0d'
EXPECTED_SHA256 = {'corpus.parquet': 'b6e16a4c45a50de3eedcde138bd33404591497c696c17f4836c4f3e9fac161ba', 'topics.miracl-v1.0-ru-train.tsv': '19c64bbd6cc0a689dbece3a684e8ab87e42e20aa2b0897fdc552235090d95628', 'qrels.miracl-v1.0-ru-train.tsv': 'bf1f737cda0d66bc38fef5f9d91843f7a89428c5c5a8a3dce4764f527ec344ef', 'topics.miracl-v1.0-ru-dev.tsv': '91a7c3b6d21f79a12b5330f0a62811962f9370a0bf539cba8b3fecaa095f0616', 'qrels.miracl-v1.0-ru-dev.tsv': 'f877b43c7f53b76aab25e406e5a607ef1c5adba4ffaa11e1b252a4326735a19b'}


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def shingles(text):
    tokens = re.findall(r'\w+', normalized(text))
    return {' '.join(tokens[i:i + 3]) for i in range(max(1, len(tokens) - 2))}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def quantiles(values):
    ordered = sorted(values)
    return {key: ordered[round((len(ordered) - 1) * p)] for key, p in [('min', 0), ('p25', .25), ('median', .5), ('p75', .75), ('p95', .95), ('max', 1)]}


def source_urls():
    urls = {'corpus.parquet': f'https://huggingface.co/datasets/sohey1024/miracl-ru-corpus/resolve/{MIRROR_REVISION}/data/train-00000-of-00001.parquet'}
    for split in ('train', 'dev'):
        for kind in ('topics', 'qrels'):
            name = f'{kind}.miracl-v1.0-ru-{split}.tsv'
            urls[name] = f'https://huggingface.co/datasets/miracl/miracl/resolve/{OFFICIAL_REVISION}/miracl-v1.0-ru/{kind}/{name}'
    return urls


class LeakageIndex:
    """Exact cross-split check and exhaustive inverted-index 3-shingle Jaccard."""
    def __init__(self):
        self.docs = {}
        self.texts = {}
        self.entries = []
        self.postings = defaultdict(list)
        self.queries = []

    def check(self, record, split):
        query_tokens = set(re.findall(r'\w+', normalized(record['query'])))
        for old_split, qid, old_query, old_tokens in self.queries:
            if old_split == split:
                continue
            jaccard = len(query_tokens & old_tokens) / max(1, len(query_tokens | old_tokens))
            if normalized(record['query']) == old_query or jaccard >= THRESHOLD:
                return {'reason': 'query_exact_or_near', 'conflict_qid': qid, 'jaccard': jaccard}
        for doc in record['docs']:
            if doc['id'] in self.docs and self.docs[doc['id']][0] != split:
                return {'reason': 'doc_id', 'docid': doc['id'], 'conflict_qid': self.docs[doc['id']][1]}
            text = normalized(doc['text'])
            if text in self.texts and self.texts[text][0] != split:
                return {'reason': 'exact_text', 'docid': doc['id'], 'conflict_qid': self.texts[text][1]}
            doc_shingles = shingles(doc['text'])
            counts = Counter(index for shingle in doc_shingles for index in self.postings.get(shingle, ()))
            for index, common in counts.items():
                old_split, old_qid, old_id, old_shingles = self.entries[index]
                if old_split != split and common / (len(doc_shingles) + len(old_shingles) - common) >= THRESHOLD:
                    return {'reason': 'near_text', 'docid': doc['id'], 'conflict_docid': old_id, 'conflict_qid': old_qid}
        return None

    def add(self, record, split):
        self.queries.append((split, record['qid'], normalized(record['query']), set(re.findall(r'\w+', normalized(record['query'])))))
        for doc in record['docs']:
            self.docs[doc['id']] = (split, record['qid'])
            self.texts[normalized(doc['text'])] = (split, record['qid'])
            ds = shingles(doc['text'])
            index = len(self.entries)
            self.entries.append((split, record['qid'], doc['id'], ds))
            for shingle in ds:
                self.postings[shingle].append(index)


def spotcheck(raw, corpus):
    revision = requests.get('https://huggingface.co/api/datasets/miracl/miracl-corpus', timeout=60).json()['sha']
    assert revision == OFFICIAL_CORPUS_REVISION, 'Official corpus revision changed; review before checking'
    results = []
    for offset in (0, 100, 1000, 10000):
        response = requests.get('https://datasets-server.huggingface.co/rows', params={'dataset': 'miracl/miracl-corpus', 'config': 'ru', 'split': 'train', 'offset': offset, 'length': 100}, timeout=60)
        response.raise_for_status()
        data = response.json()
        matches = []
        for item in data['rows']:
            row = item['row']
            if row['docid'] in corpus:
                mirror = corpus[row['docid']]
                assert row['title'] == mirror['title'] and row['text'] == mirror['text'], row['docid']
                matches.append(row['docid'])
        results.append({'offset': offset, 'status': response.status_code, 'returned': len(data['rows']), 'mirror_exact_matches': matches})
    assert sum(len(r['mirror_exact_matches']) for r in results) > 0
    write_json(raw / 'official_rows_spotcheck.json', results)


def prepare(root, check_official=False):
    raw, prepared = root / 'data/raw', root / 'data/prepared'
    raw.mkdir(parents=True, exist_ok=True)
    prepared.mkdir(parents=True, exist_ok=True)
    sources = []
    for name, url in source_urls().items():
        target = raw / name
        if not target.exists():
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            target.write_bytes(response.content)
        assert digest(target) == EXPECTED_SHA256[name], f'Corrupt source file: {target}'
        sources.append({'file': name, 'url': url, 'bytes': target.stat().st_size, 'sha256': digest(target)})
    rows = pq.read_table(raw / 'corpus.parquet').to_pylist()
    corpus = {r['doc_id']: {'id': r['doc_id'], 'title': r['title'], 'text': r['text']} for r in rows}
    assert len(corpus) == len(rows), 'Duplicate corpus doc ids'
    assert all(d['text'].strip() for d in corpus.values()), 'Empty corpus text'
    if check_official:
        spotcheck(raw, corpus)
    spotcheck_path = raw / 'official_rows_spotcheck.json'
    official_spotcheck = {'status': 'not_run'}
    if spotcheck_path.exists():
        checks = json.loads(spotcheck_path.read_text(encoding='utf-8'))
        official_spotcheck = {'status': 'passed', 'source': 'https://datasets-server.huggingface.co/rows?dataset=miracl/miracl-corpus&config=ru&split=train', 'corpus_revision_at_check': OFFICIAL_CORPUS_REVISION, 'rows_checked': sum(r['returned'] for r in checks), 'exact_title_and_text_matches': sum(len(r['mirror_exact_matches']) for r in checks), 'log_sha256': digest(spotcheck_path), 'matched_docids': [docid for r in checks for docid in r['mirror_exact_matches']]}

    full_records, official_stats = {}, {}
    for source_split in ('train', 'dev'):
        topics = dict(line.split('\t', 1) for line in (raw / f'topics.miracl-v1.0-ru-{source_split}.tsv').read_text(encoding='utf-8').splitlines())
        qrels = defaultdict(dict)
        for line in (raw / f'qrels.miracl-v1.0-ru-{source_split}.tsv').read_text(encoding='utf-8').splitlines():
            qid, _, docid, rel = line.split('\t')
            assert docid not in qrels[qid], 'Duplicate qrel'
            qrels[qid][docid] = int(rel)
        full_records[source_split] = []
        missing = []
        for qid in sorted(topics, key=int):
            if any(docid not in corpus for docid in qrels[qid]):
                missing.append(qid)
                continue
            full_records[source_split].append({'qid': source_split + ':' + qid, 'official_qid': qid, 'official_split': source_split, 'query': topics[qid], 'docs': [{**corpus[docid], 'rel': rel} for docid, rel in sorted(qrels[qid].items())]})
        rels = [r for labels in qrels.values() for r in labels.values()]
        official_stats[source_split] = {'queries': len(topics), 'judgments': len(rels), 'positives': sum(r > 0 for r in rels), 'negatives': sum(r == 0 for r in rels), 'missing_document_queries': missing, 'covered_judgments': sum(docid in corpus for labels in qrels.values() for docid in labels), 'judgments_per_query': quantiles(list(map(len, qrels.values())))}
    selected, excluded = {}, []
    index = LeakageIndex()
    eligible_stats = {}
    for source_split, targets in [('train', ('train', 'validation')), ('dev', ('test', 'application'))]:
        pool = list(full_records[source_split])
        random.Random(SEED + (0 if source_split == 'train' else 1)).shuffle(pool)
        eligible_stats[source_split] = sum(1 for r in pool if 2 <= len(r['docs']) <= 20 and any(d['rel'] > 0 for d in r['docs']) and any(d['rel'] == 0 for d in r['docs']))
        cursor = 0
        for split in targets:
            selected[split] = []
            while len(selected[split]) < SIZES[split] and cursor < len(pool):
                record = pool[cursor]
                cursor += 1
                if not (2 <= len(record['docs']) <= 20 and any(d['rel'] > 0 for d in record['docs']) and any(d['rel'] == 0 for d in record['docs'])):
                    excluded.append({'qid': record['qid'], 'intended_split': split, 'reason': 'requires_positive_and_negative_and_at_most_20_judgments'})
                    continue
                conflict = index.check(record, split)
                if conflict:
                    excluded.append({'qid': record['qid'], 'intended_split': split, **conflict})
                    continue
                selected[split].append(record)
                index.add(record, split)
            assert len(selected[split]) == SIZES[split], f'Insufficient eligible queries: {split}'
            write_jsonl(prepared / f'{split}.jsonl', selected[split])
    write_jsonl(prepared / 'corpus.jsonl', sorted(corpus.values(), key=lambda d: d['id']))
    write_json(prepared / 'exclusions.json', excluded)
    split_stats = {}
    for split, records in selected.items():
        docs = [d for r in records for d in r['docs']]
        positives = sum(d['rel'] > 0 for d in docs)
        split_stats[split] = {'queries': len(records), 'judgments': len(docs), 'positive_judgments': positives, 'negative_judgments': len(docs) - positives, 'positive_fraction': positives / len(docs), 'unique_docids': len({d['id'] for d in docs}), 'passage_characters': quantiles([len(d['text']) for d in docs]), 'query_characters': quantiles([len(r['query']) for r in records]), 'judgments_per_query': quantiles([len(r['docs']) for r in records]), 'qids': [r['qid'] for r in records]}
    overlaps = []
    for i, a in enumerate(SIZES):
        for b in list(SIZES)[i + 1:]:
            da, db = [d for r in selected[a] for d in r['docs']], [d for r in selected[b] for d in r['docs']]
            overlaps.append({'splits': [a, b], 'docid_overlap': len({d['id'] for d in da} & {d['id'] for d in db}), 'normalized_text_overlap': len({normalized(d['text']) for d in da} & {normalized(d['text']) for d in db})})
    assert all(row['docid_overlap'] == row['normalized_text_overlap'] == 0 for row in overlaps)
    normalized_counts = Counter(normalized(d['text']) for d in corpus.values())
    passport = {'dataset': 'MIRACL-ru judged-passage subset', 'language': 'Russian', 'official_topics_qrels_revision': OFFICIAL_REVISION, 'text_mirror': 'sohey1024/miracl-ru-corpus', 'text_mirror_revision': MIRROR_REVISION, 'sources': sources, 'official_text_spotcheck': official_spotcheck, 'seed': SEED, 'selection': 'Sorted official integer qids, shuffle train seed42 / dev seed43, take first eligible queries. Train and validation from official train; test and application from official dev. Validation is reserved for model selection. Cross-split conflicts discard complete queries. All official judgments for every retained query are preserved.', 'eligibility': 'At least one positive and one negative judgment; 2..20 total judgments. No filtering by model success, text length or BM25 difficulty.', 'qid_namespace': 'official_split:official_qid because original train/dev numeric qids overlap', 'near_duplicates': {'text_method': 'Exhaustive cross-split token 3-shingle Jaccard via inverted index; NFKC casefold whitespace normalization.', 'query_method': 'Normalized exact text or word-set Jaccard.', 'jaccard_threshold': THRESHOLD, 'cross_split_near_overlap_after_filter': 0}, 'official_statistics': official_stats, 'eligible_queries': eligible_stats, 'corpus': {'passages': len(corpus), 'unique_docids': len(corpus), 'normalized_duplicate_text_groups': sum(c > 1 for c in normalized_counts.values()), 'extra_duplicate_text_passages': sum(c - 1 for c in normalized_counts.values() if c > 1), 'characters': quantiles([len(d['text']) for d in corpus.values()]), 'scope': 'Fixed global union of 41007 passages judged for official Russian train/dev. No per-query insertion. NOT the full 22-million-passage MIRACL-ru collection.', 'construction_bias': 'Corpus inclusion depends on historical relevance-annotation pools, including held-out queries. This makes a closed judged-pool retrieval benchmark easier than full-corpus retrieval. Relevance judgments remain incomplete for other queries.'}, 'splits': split_stats, 'cross_split_checks': overlaps, 'excluded_queries': len(excluded), 'exclusion_reasons': dict(Counter(row['reason'] for row in excluded)), 'preprocessing': 'Original titles/texts retained verbatim. No truncation in stored data. Normalization used only for leakage checks. Runner separately limits model-visible text.', 'limitations': ['MIRACL questions are single-turn retrieval questions, without multi-hop gold trajectories.', 'Restricted global corpus uses judged-pool texts; results must not be compared directly to full MIRACL leaderboard.', 'Binary qrels have incomplete judgment coverage. Treating unjudged retrieved texts as nonrelevant biases recall/nDCG.', 'Text source is a third-party mirror; official docid/qrel coverage is fully checked. Official text spot checks depend on source-server availability.'], 'prepared_sha256': {p.name: digest(p) for p in prepared.glob('*.jsonl')}}
    passport.update({'unit': 'query and judged Wikipedia passage; binary relevance 0/1',
        'license': {'official_card': 'Apache-2.0', 'mirror_card': 'not specified; upstream corpus is MIRACL/Wikipedia',
                    'official_url': 'https://huggingface.co/datasets/miracl/miracl',
                    'corpus_url': 'https://huggingface.co/datasets/miracl/miracl-corpus'},
        'temporal_structure': 'No per-query event/publication timestamp; random grouped split, not temporal forecasting.',
        'missing_values': {'empty_text': sum(not d['text'].strip() for d in corpus.values()),
                           'empty_title': sum(not d['title'].strip() for d in corpus.values()),
                           'empty_selected_query': sum(not r['query'].strip() for rows in selected.values() for r in rows)}})
    write_json(prepared / 'passport.json', passport)
    (root / 'results').mkdir(exist_ok=True)
    write_json(root / 'results/data_audit.json', {'status': 'passed', 'official_coverage': official_stats, 'splits': split_stats, 'cross_split_checks': overlaps, 'near_duplicate_check': passport['near_duplicates'], 'source_checksums': sources, 'official_text_spotcheck': official_spotcheck, 'exclusions': passport['exclusion_reasons']})
    print(json.dumps({'corpus': len(corpus), 'splits': {s: {'queries': v['queries'], 'judgments': v['judgments']} for s, v in split_stats.items()}, 'exclusions': passport['exclusion_reasons']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spotcheck', action='store_true', help='Recheck sampled mirror texts against the official server')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    prepare(args.root, args.spotcheck)
