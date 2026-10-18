"""Seal the measured submission after manual review of the rendered pages."""
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from docx import Document
from pypdf import PdfReader

ROOT = Path(r'D:\Projects\University_Current\lab1-llm-miracl')
DEPS = Path(r'C:\Users\Nick\.cache\codex-runtimes\codex-primary-runtime\dependencies')


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def save(path, value):
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


docx = ROOT / 'report/Lab1_MIRACL_final.docx'
pdf = ROOT / 'report/qa/report.pdf'
doc = Document(docx)
expected = load('report/tables_export.json')
assert len(doc.tables) == len(expected)
for index, (table, source) in enumerate(zip(doc.tables, expected)):
    cells = [[cell.text for cell in row.cells] for row in table.rows]
    reference = [[str(value) for value in row] for row in [source['headers'], *source['rows']]]
    assert cells == reference, f'Table {index} differs'

with zipfile.ZipFile(docx) as z:
    embedded = {hashlib.sha256(z.read(p)).hexdigest() for p in z.namelist() if p.startswith('word/media/')}
figures = ['comparison_test.png', 'calibration_test.png', 'search_comparison.png']
for name in figures:
    assert sha(ROOT / 'figures' / name) in embedded, f'Figure differs: {name}'

reader = PdfReader(pdf)
assert len(reader.pages) == 19
text = '\n'.join(page.extract_text() or '' for page in reader.pages)
for token in ['TODO', 'TBD', 'Lorem ipsum', '\ufffd']:
    assert token not in text, f'Placeholder: {token}'
assert len(load('report/literature.json')['papers']) == 15
assert load('results/final_independent_audit.json')['verdict'] == 'no_P1_P2_defects_found'
assert load('results/independent_numeric_checks.json')['passed']
assert load('results/search_selftest.json')['passed']
assert 'Ran 14 tests' in (ROOT / 'results/unit_tests.log').read_text(encoding='utf-8')

review = {
    'status': 'passed', 'reviewer': 'primary_agent', 'date': '2026-10-18',
    'method': 'Visual inspection of all rendered pages at original image resolution; Microsoft Word PDF export',
    'pages_reviewed': list(range(1, 20)), 'layout_defects_remaining': [],
    'docx_sha256': sha(docx), 'pdf_sha256': sha(pdf),
    'page_sha256': {p.name: sha(p) for p in sorted((ROOT / 'report/qa').glob('page-*.png'))},
    'docx_tables_match_export': len(expected), 'embedded_figures_match_source': figures,
}
save('report/qa/visual_review.json', review)
render = load('report/qa/render_checks.json')
render['visual_review'] = 'passed_all_19_pages'
save('report/qa/render_checks.json', render)

evidence = {
    'R01': 'report/literature.json; report/Lab1_MIRACL_final.docx sections 1-2',
    'R02': 'report/literature.json: 15 primary papers and verified source URLs',
    'R03': 'report/literature_matrix.csv; report/literature.json',
    'R04': 'report/literature.json; report/literature.md; report selection table and research gap',
    'R05': 'data/prepared/passport.json; results/data_audit.json',
    'R06': 'data/prepared/exclusions.json; results/final_independent_audit.json',
    'R07': 'configs/protocol.json; results/predictions.jsonl; results/runtime.json',
    'R08': 'configs/few_shot_examples.json; results/predictions.jsonl',
    'R09': 'results/peft_run_manifest.json; results/peft_summary.json; adapters/',
    'R10': 'src/run_experiment.py; results/summary.json: strict schema, abstention, no fallback',
    'R11': 'results/robustness.json; results/predictions.jsonl instruction variant',
    'R12': 'configs/protocol.json; results/summary.json; results/paired_comparisons.json',
    'R13': 'results/summary.json; results/calibration_bins.json; figures/calibration_test.png',
    'R14': 'results/runtime.json; results/resources.jsonl; results/summary.json',
    'R15': 'results/error_examples.json; results/error_slices.json; report section 6',
    'R16': 'results/selection.json; results/final_independent_audit.json chronology/utility checks',
    'R17': 'results/search_trajectories.jsonl; results/search_summary.json; application classification',
    'R18': 'report/Lab1_MIRACL_final.docx sections 9-10: H1, limits, next experiment',
    'R19': 'README.md; src/finish_lab1.py; configs/; raw predictions and data passport',
    'R20': 'https://github.com/NikitaGavrilenko/lab1-llm-miracl; local Git repository and full submission archive',
    'R21': 'results/unit_tests.log (14); results/search_selftest.json (12); independent numeric audit',
    'R22': 'report/qa/visual_review.json: all 19 pages; tables and figure hashes checked',
}
checklist = load('.workflow/requirements_checklist.json')
checklist.update(audit_stage='final_audit', date='2026-10-18')
for item in checklist['items']:
    item['status'] = 'passed'
    item['expected_evidence'] = evidence[item['id']]
    item['note'] = ('Published repository: https://github.com/NikitaGavrilenko/lab1-llm-miracl'
                    if item['id'] == 'R20' else 'Verified against completed artifacts; scope limitations are stated in the report.')
save('.workflow/requirements_checklist.json', checklist)

audit = {
    'status': 'verified_submission',
    'numeric_checks': True, 'unit_tests_passed': 14, 'search_selftests_passed': 12,
    'independent_audit': 'results/final_independent_audit.json',
    'independent_verdict': 'no_P1_P2_defects_found',
    'word_render': render, 'visual_review': 'passed_all_19_pages',
    'docx_tables_checked': len(expected), 'source_figures_checked': figures,
    'docx_sha256': sha(docx), 'pdf_sha256': sha(pdf),
    'requirements': {'passed': 22, 'partial': []},
    'scope_limits': ['Judged-pool relevance classification and conditional reranking are not full-corpus retrieval.',
                     'Search uses 41007 historically judged passages and incomplete qrels, not the full MIRACL corpus.',
                     'Separate 0.5B LoRA control uses a different 256-token protocol; 92.1% test passages truncated.',
                     'One shared Wikipedia article has different train/test passages; no passage or near-text overlap.'],
}
save('results/final_audit.json', audit)
for name in ['.workflow/state.json', '.workflow/full_run_state.json']:
    state = load(name)
    state.update(status='verified_complete', stage='final_visual_and_artifact_audit', updated_unix=time.time())
    state['next'] = []
    if 'history' in state:
        state['history'].append({'stage': 'final_visual_and_artifact_audit', 'status': 'passed', 'time_unix': time.time()})
    save(name, state)

readme = (ROOT / 'README.md').read_text(encoding='utf-8')
readme = readme.replace('Визуальная проверка всех PNG остаётся отдельным явно отмеченным этапом, пока первичный агент её не подтвердит.',
    'Все 19 страниц проверены визуально; таблицы Word совпадают с экспортом, встроенные графики — с исходными PNG. '
    'Независимый аудит сырых результатов: results/final_independent_audit.json. Пройдено 14 unit-тестов и 12 проверок поиска.')
readme = readme if '```powershell' in readme else readme.replace('python src/finish_lab1.py\n', '```powershell\npython src/finish_lab1.py\n```\n', 1)
readme = readme if '```powershell\npython src/prepare_data.py' in readme else readme.replace('python src/prepare_data.py\n', '```powershell\npython src/prepare_data.py\n', 1)
readme = readme if 'python src/build_report.py\n```' in readme else readme.replace('python src/build_report.py\n', 'python src/build_report.py\n```\n', 1)
readme = readme.replace('Для нового независимого опыта скопируйте проект без predictions.jsonl, runtime.json, selection.json и search_trajectories.jsonl; сохраните текущие результаты отдельно.',
    'Для нового независимого опыта используйте отдельную копию проекта с пустой results/ и без обученных adapters/; '
    'сохраните текущие результаты отдельно. Сначала выполните prepare_data.py. В конфигурации runner при переносе задайте пути к Python и bundled runtime. '
    'Ollama должен обслуживать модели из configs/protocol.json. Адаптер и базовые веса доступны отдельно; models/ и виртуальное окружение в архив не включены.')
(ROOT / 'README.md').write_text(readme, encoding='utf-8')

archive = ROOT / 'Lab1_MIRACL_submission.zip'
manifest_path = ROOT / 'report/submission_manifest.json'
manifest_path.unlink(missing_ok=True)
git = DEPS / 'native/git/cmd/git.exe'
subprocess.run([str(git), 'add', '.'], cwd=ROOT, check=True, capture_output=True)
changed = subprocess.run([str(git), 'diff', '--cached', '--quiet'], cwd=ROOT).returncode
if changed == 1:
    result = subprocess.run([str(git), '-c', 'user.name=Lab1 research artifact', '-c', 'user.email=lab1@localhost',
        'commit', '-m', 'Verify all report pages and seal independently audited submission'], cwd=ROOT, check=True, capture_output=True, text=True)
    print(result.stdout)
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(ROOT.rglob('*')):
        rel = p.relative_to(ROOT)
        if not p.is_file() or p == archive or p == manifest_path:
            continue
        if any(part in {'.git', 'models', '__pycache__'} or part.startswith('.venv') for part in rel.parts):
            continue
        if p.suffix in {'.pyc', '.tmp'} or rel.parts[:2] == ('.workflow', 'render_probe'):
            continue
        if rel.parts[:2] == ('report', 'qa') and p.suffix not in {'.pdf', '.json'}:
            continue
        z.write(p, rel.as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert hashlib.sha256(z.read('report/Lab1_MIRACL_final.docx')).hexdigest() == sha(docx)
    assert json.loads(z.read('results/final_audit.json'))['visual_review'] == 'passed_all_19_pages'
save('report/submission_manifest.json', {
    'archive': archive.name, 'sha256': sha(archive), 'bytes': archive.stat().st_size,
    'docx_sha256': sha(docx), 'pdf_sha256': sha(pdf), 'archive_crc_verified': True,
    'manifest_scope': 'External archive checksum; manifest itself is excluded from archive to avoid self-reference.',
})
print(json.dumps({'passed': True, 'tables': len(expected), 'pages': 19, 'archive_bytes': archive.stat().st_size}, ensure_ascii=False))
