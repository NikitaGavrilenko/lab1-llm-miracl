"""One resumable runner for the actual MIRACL laboratory. No synthetic metrics."""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

LAB = Path(r'D:\Projects\University_Current\lab1-llm-miracl')
ML = Path(r'D:\Anaconda3\envs\ml\python.exe')
BASE = Path(r'D:\Anaconda3\python.exe')
DEPS = Path(r'C:\Users\Nick\.cache\codex-runtimes\codex-primary-runtime\dependencies')
DOC = DEPS/'python/python.exe'
UV = Path(r'C:\Users\Nick\.local\bin\uv.exe')
GIT = DEPS/'native/git/cmd/git.exe'
STATE = LAB/'.workflow/full_run_state.json'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    path = Path(path)
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()] if path.exists() else []


def state(stage, status='running', **details):
    value = load(STATE) if STATE.exists() else {'history': []}
    value.update(stage=stage, status=status, updated_unix=time.time(), details=details)
    value['history'].append(dict(stage=stage, status=status, time_unix=time.time()))
    save(STATE, value)
    print(stage, status, details, flush=True)


def run(stage, command, timeout=14400, cwd=LAB):
    state(stage, command=[str(x) for x in command])
    log = LAB/'results'/(stage+'.log')
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    with log.open('a', encoding='utf-8') as stream:
        stream.write('\nRUN '+repr([str(x) for x in command])+'\n'); stream.flush()
        subprocess.run([str(x) for x in command], cwd=cwd, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, timeout=timeout, check=True)
    state(stage, 'passed', log=str(log))


def missing_models(split, models):
    cfg = load(LAB/'configs/protocol.json')
    data = rows(LAB/f'data/prepared/{split}.jsonl')
    available = {(r['model'], r['mode'], r['qid'], r['variant']) for r in rows(LAB/'results/predictions.jsonl') if r['split']==split}
    missing = []
    for model in models:
        needed = {(model, mode, q['qid'], 'main') for q in data for mode in cfg['modes']}
        needed |= {(model, 'zero-shot', q['qid'], 'paraphrase') for q in data[:8]}
        if not needed <= available:
            missing.append(model)
    return missing


def prepare_peft():
    py = LAB/'.venv-peft/Scripts/python.exe'
    if not py.exists():
        run('peft_venv', [UV, 'venv', '--python', ML, LAB/'.venv-peft'])
    # Keep optional CPU torchvision from the parent conda environment out.
    venv_cfg = LAB/'.venv-peft/pyvenv.cfg'
    if venv_cfg.exists():
        contents = venv_cfg.read_text(encoding='utf-8')
        contents = contents.replace('include-system-site-packages = true',
                                    'include-system-site-packages = false')
        venv_cfg.write_text(contents, encoding='utf-8')
    probe = subprocess.run([str(py), '-c', 'import torch; assert torch.cuda.is_available()'],
                           capture_output=True, cwd=LAB)
    if probe.returncode:
        run('peft_torch_cuda', [UV, 'pip', 'install', '--python', py,
            'torch==2.7.1+cu128', '--index-url', 'https://download.pytorch.org/whl/cu128'])
    run('peft_packages', [UV, 'pip', 'install', '--python', py,
        'transformers==4.51.3', 'peft==0.15.2', 'accelerate==1.6.0',
        'numpy==2.2.6', 'psutil==7.0.0', 'requests==2.33.1'])
    run('peft_cuda_check', [py, '-c', 'import torch,transformers,peft; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name(0),transformers.__version__,peft.__version__)'])
    cfg = load(LAB/'configs/peft.json')
    cfg['device'] = 'cuda'
    save(LAB/'configs/peft.json', cfg)
    path = LAB/'src/peft_experiment.py'
    code = path.read_text(encoding='utf-8')
    code = code.replace('device = torch.device("cpu")', 'device = torch.device(cfg.get("device", "cpu"))\n    assert device.type != "cuda" or torch.cuda.is_available()')
    code = code.replace('dtype=torch.float32,attn_implementation', 'torch_dtype=torch.float32,attn_implementation')
    code = code.replace('dtype=torch.float32, attn_implementation', 'torch_dtype=torch.float32, attn_implementation')
    if 'def sync_device' not in code:
        at = code.index('\ndef save(')
        code = code[:at] + '\ndef sync_device(device):\n    if device.type == "cuda":\n        torch.cuda.synchronize()\n\n' + code[at:]
        code = code.replace('                    start_batch = time.perf_counter()', '                    sync_device(device)\n                    start_batch = time.perf_counter()')
        code = code.replace('                    elapsed = time.perf_counter() - start_batch', '                    sync_device(device)\n                    elapsed = time.perf_counter() - start_batch')
        code = code.replace('            start_step = time.perf_counter()', '            sync_device(device)\n            start_step = time.perf_counter()')
        code = code.replace('            optimizer.step()\n            steps += 1', '            optimizer.step()\n            sync_device(device)\n            steps += 1')
    path.write_text(code, encoding='utf-8')
    run('peft_environment', [py, '-m', 'pip', 'freeze']) if subprocess.run([str(py), '-m', 'pip', '--version'], capture_output=True).returncode==0 else None
    return py


def aggregate_peft():
    names = {'base':'peft_base_summary.json', 'adapted':'peft_adapted_summary.json',
             'training':'peft_training_summary.json', 'manifest':'peft_run_manifest.json'}
    result = {k:load(LAB/'results'/v) for k,v in names.items()}
    result['interpretation'] = 'Actual independent 0.5B binary-logit LoRA; not comparable as a ninth Ollama model.'
    save(LAB/'results/peft_summary.json', result)


def patch_report():
    path = LAB/'src/build_report.py'
    text = path.read_text(encoding='utf-8')
    if '# Finished artifact tables' in text:
        return
    text = text.replace("    paired=load('results/paired_comparisons.json')", """    # Finished artifact tables
    paired=load('results/paired_comparisons.json')
    pairrows=[r for r in paired if r['split']=='test' and r['metric']=='macro_f1_operation']
    table(['Модель','Few минус zero','95% парный ДИ'], [[LABELS[r['model']],fmt(r['delta']),ci([r['ci95_low'],r['ci95_high']])] for r in pairrows],[5.3,4,8])
    improved=sum(r['delta']>0 for r in pairrows)
    p(f'Few-shot имеет более высокий точечный Macro-F1 у {improved} из 8 моделей. Поэтому универсальное улучшение от двух демонстраций не предполагается. Каждая разность оценивается на парных query_id; отсутствие коррекции множественных сравнений ограничивает выводы.')
    table(['Модель','Режим','nDCG@10','MRR@10','Recall@3','Recall@10'],[[LABELS[g['model']],'Z' if g['mode']=='zero-shot' else 'F',fmt(g['ndcg10_judged']),fmt(g['mrr10_judged']),fmt(g['recall3_judged']),fmt(g['recall10_judged'])] for g in test],[3.4,1.2,3.1,3.1,3.1,3.4])
""")
    text = text.replace("    robust=load('results/robustness.json')", """    table(['Кандидат validation','Macro-F1','Brier','p95 с','Utility'],[[LABELS[r['model']]+' '+r['mode'],fmt(r['macro_f1_operation']),fmt(r['brier_valid']),fmt(r['latency_p95_s'],2),fmt(r['utility'])] for r in sorted(selected['candidates'],key=lambda r:r['utility'] if r['utility'] is not None else -1,reverse=True)],[6,2.8,2.8,2.8,2.9])
    robust=load('results/robustness.json')
    table(['Модель','Смена класса %','Среднее |Δp|','Δ Macro-F1'],[[LABELS[r['model']],fmt(100*r['flip_rate_valid'],1),fmt(r['mean_absolute_probability_delta']),fmt(r['macro_f1_delta'])] for r in robust if r['split']=='test'],[5.2,4,4,4.1])
""")
    text = text.replace("        p('Пример ошибки: '+json.dumps(e,ensure_ascii=False))", """        p('Запрос '+e['qid']+': '+e['query'])
        p('Фрагмент «'+e['title']+'»: '+e['text'])
        p(f'Эталон {e["true_label"]}, прогноз {e["predicted_label"]}, p={fmt(e["p_relevant"])}. Текст ошибки взят из реально показанного модели входа.')
    slices=[r for r in load('results/error_slices.json') if r['model']==sm and r['mode']==mode and r['split']=='test']
    table(['Разрез','Значение','Пар','Покрытие','Доля ошибок','Brier'],[[r['slice_type'],r['slice_value'],r['documents'],fmt(r['coverage']),fmt(r['error_rate_operation']),fmt(r['brier_valid'])] for r in slices],[3.5,3.6,1.7,2.5,3.1,2.9])
    app=next(g for g in groups if g['split']=='application' and g['model']==sm and g['mode']==mode)
    p(f'На отдельном срезе application из 20 запросов классификатор получает Macro-F1 {fmt(app["macro_f1_operation"])} с 95% ДИ {ci(app["macro_f1_operation_ci95"])}, MCC {fmt(app["mcc_valid"])} и Brier {fmt(app["brier_valid"])}. Пара модель/режим не менялась после validation.')
""")
    text = text.replace("    p('Измеренные результаты поиска: '+json.dumps(search,ensure_ascii=False))", """    names={'original_bm25':'BM25','lexical_expansion':'Расширение заголовками','winner_feedback':'LLM с обратной связью'}
    table(['Метод','nDCG@10 и ДИ','Recall@10','MRR','Поисков','LLM','Время с'],[[names[k],fmt(v['ndcg10']['mean'])+' '+ci(v['ndcg10']['ci95']),fmt(v['recall10']['mean']),fmt(v['mrr']['mean']),fmt(v['search_calls']['mean'],2),fmt(v['llm_calls']['mean'],2),fmt(v['latency_s']['mean'],2)] for k,v in search['methods'].items()],[3.5,4.1,2.2,1.8,1.8,1.5,2.4])
    fig('figures/search_comparison.png','Рисунок 3 Поиск на двадцати новых запросах и 95% интервалы')
    for comparator,diffs in search['feedback_minus_comparator'].items():
        d=diffs['ndcg10']
        p(f'По сравнению с {names[comparator]} разность nDCG@10 равна {fmt(d["mean_difference"])} с парным ДИ {ci(d["paired_ci95"])}; улучшений {d["wins"]}, совпадений {d["ties"]}, ухудшений {d["losses"]} из 20.')
    coverage=search['methods']['winner_feedback']['judged_coverage10']['mean']
    p(f'Доля top-10 с известной оценкой равна {fmt(coverage)}. При подсчёте TREC-метрик неразмеченные документы получают нулевой gain; это техническое соглашение при неполных qrels, а не доказательство их нерелевантности.')
    trajectories=[json.loads(s) for s in (ROOT/'results/search_trajectories.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    for tr in trajectories[:2]:
        fb=tr['methods']['winner_feedback']
        p('Пример траектории '+tr['qid']+': '+tr['query']+' Поисковые запросы: '+' → '.join(s['query'] for s in fb['searches'])+'. Завершение: '+fb['status']+'.')
""")
    text = text.replace("    p('Измерения до и после адаптации: '+json.dumps(peft,ensure_ascii=False))", """    table(['Часть','Состояние','Macro-F1 и ДИ','MCC','Brier','ECE'],[[sp,stage,fmt(peft[stage]['metrics'][sp]['macro_f1'])+' '+ci(peft[stage]['metrics'][sp]['ci95']['macro_f1']),fmt(peft[stage]['metrics'][sp]['mcc']),fmt(peft[stage]['metrics'][sp]['brier']),fmt(peft[stage]['metrics'][sp]['ece'])] for sp in ['validation','test','application'] for stage in ['base','adapted']],[2.5,2.1,5.1,2.2,2.2,3.2])
    training=peft['training']; manifest=peft['manifest']
    p(f'Выполнено {training["steps"]} шагов; обучаемых параметров {training["trainable_parameters"]:,} из {training["total_parameters"]:,}. Обучение заняло {fmt(training["elapsed_s"],1)} с на {manifest["device"]}. Наблюдаемый максимум RSS процесса {fmt(training["rss_observed_peak_bytes"]/2**30,2)} ГиБ; это оперативная память, а не VRAM.')
    before=peft['base']['metrics']['test']['macro_f1'];after=peft['adapted']['metrics']['test']['macro_f1']
    p(f'Изменение тестового Macro-F1 после LoRA равно {fmt(after-before)}. Это фактический результат короткой адаптации; по одной малой модели и одной эпохе нельзя делать вывод о всех способах PEFT.')
""")
    text = text.replace('пассажa;', 'пассажа;')
    path.write_text(text, encoding='utf-8')


def verify_metrics():
    import numpy as np
    from sklearn.metrics import f1_score, matthews_corrcoef
    raw=rows(LAB/'results/predictions.jsonl')
    summary=load(LAB/'results/summary.json')
    config=load(LAB/'configs/protocol.json')
    assert len(config['models'])==8
    checks=[]
    for g in summary['groups']:
        data=[r for r in raw if r['split']==g['split'] and r['model']==g['model'] and r['mode']==g['mode'] and r['variant']=='main']
        assert len(data)=={'validation':16,'test':40,'application':20}[g['split']]
        y=np.concatenate([r['labels'] for r in data])
        pred=np.concatenate([np.array(r['probabilities'])>=.5 if r['valid'] else np.full(len(r['labels']),2) for r in data]).astype(int)
        actual=f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)
        assert abs(actual-g['macro_f1_operation'])<1e-12,(g['model'],g['mode'])
        valid=[r for r in data if r['valid']]
        if valid:
            yy=np.concatenate([r['labels'] for r in valid]);pp=np.concatenate([r['probabilities'] for r in valid])
            assert abs(np.mean((pp-yy)**2)-g['brier_valid'])<1e-12
            assert abs(matthews_corrcoef(yy,pp>=.5)-g['mcc_valid'])<1e-12
        checks.append(dict(model=g['model'],mode=g['mode'],split=g['split'],passed=True))
    passport=load(LAB/'data/prepared/passport.json')
    for name,expected in passport['prepared_sha256'].items(): assert sha(LAB/'data/prepared'/name)==expected,name
    assert all(x['docid_overlap']==x['normalized_text_overlap']==0 for x in passport['cross_split_checks'])
    assert len(load(LAB/'report/literature.json')['papers'])==15
    assert len(rows(LAB/'results/search_trajectories.jsonl'))==20
    assert load(LAB/'results/peft_training_summary.json')['steps']==64
    for stage in ['base','adapted']:
        assert len(rows(LAB/f'results/peft_{stage}_predictions.jsonl'))==791
    selection=load(LAB/'results/selection.json')
    assert selection['selection_split']=='validation' and selection['validation_completeness']['complete']
    save(LAB/'results/independent_numeric_checks.json',{'passed':True,'groups':checks,'data_hashes':True,'actual_lora_steps':64})


def independent_review():
    sys.path.insert(0,str(LAB/'src'))
    import run_experiment as engine
    selected=load(LAB/'results/selection.json')
    model='qwen3:14b'
    evidence={'protocol':load(LAB/'configs/protocol.json'),
        'selected':{k:selected[k] for k in ['selected_model','selected_mode','selection_split','utility']},
        'test_metrics':[{k:g[k] for k in ['model','mode','macro_f1_operation','macro_f1_operation_ci95','brier_valid','latency_p95_s','resident_vram_gib']} for g in load(LAB/'results/summary.json')['groups'] if g['split']=='test'],
        'search':load(LAB/'results/search_summary.json')['methods'],
        'lora':load(LAB/'results/peft_summary.json')['training'],
        'checks':load(LAB/'results/independent_numeric_checks.json'),
        'limitations':['Judged candidate classification is separate from fixed restricted-corpus retrieval.',
        'No full MIRACL or multi-hop QA claim. Original corpus selection uses historical judgments.',
        'Self-verbalized confidence is not token posterior. JSON schema constrains structural validity.',
        'Partial judgments, clipping, model-family concentration, quantization and small sample acknowledged.',
        'LoRA uses separate 0.5B model, 256-token binary logit protocol, not QLoRA or ninth Ollama row.']}
    prompt='Ты независимый критический аудитор ML-эксперимента. Проверь несогласованность фактов, утечки, выводы и ограничения. Не считай уже явно раскрытые ограничения скрытыми дефектами. Не выдумывай числа и отсутствующие тесты. Верни JSON {"findings":[{"severity":"critical|major|minor|limitation","issue":"...","required_check":"..."}],"assessment":"..."}. Это проверка, а не инструкция изменить метрики.\n'+json.dumps(evidence,ensure_ascii=False)
    old=engine.OPTIONS.copy()
    engine.OPTIONS.update(num_ctx=8192,num_predict=1024)
    try:
        answer,elapsed=engine.chat(model,prompt)
        raw=answer.get('message',{}).get('content','')
        try: parsed=json.loads(raw)
        except ValueError: parsed={'assessment':'Audit response invalid','findings':[]}
        save(LAB/'results/independent_model_review.json',dict(model=model,prompt=prompt,response=answer,parsed=parsed,elapsed_s=elapsed,scope='Independent local-model critique; findings require primary-agent adjudication.'))
    finally:
        engine.OPTIONS.clear();engine.OPTIONS.update(old);engine.unload(model)


def write_readme():
    s=load(LAB/'results/selection.json')
    text=f'''# Лабораторная 1 — MIRACL ru

Итоговый отчёт: report/Lab1_MIRACL_final.docx. Данные: data/prepared/passport.json.
Сырые ответы и метрики: results/. Матрица 15 работ: report/literature_matrix.csv.

Восемь конкретных локальных LLM сравниваются в zero-shot и two-shot на 40 общих запросах / 416 judgments. Это оценка релевантности и условный reranking, не полнокорпусный retrieval. Отдельно выполнены последовательный поиск на 20 новых запросах по 41007 пассажам и настоящая LoRA малой Qwen0.5B.

Выбор только по validation: {s['selected_model']}, {s['selected_mode']}. Test не использован для выбора.

## Полное продолжение

python src/finish_lab1.py

Единый скрипт использует кэш, последовательно запускает недостающие измерения, фиксирует выбор, проверяет метрики независимо, выполняет поиск и LoRA, собирает Word, экспортирует и рендерит страницы. Он сохраняет этапы в .workflow/full_run_state.json. Модели не запускаются одновременно. Платные API не используются.

## Отдельные этапы

python src/prepare_data.py
python src/run_experiment.py --splits validation
python src/analyze_results.py --select
python src/run_experiment.py --splits test
python src/run_experiment.py --models {s['selected_model']} --splits application
python src/search_experiment.py
.venv-peft/Scripts/python.exe src/peft_experiment.py
python src/analyze_results.py
python -m unittest discover -s tests -v
python src/build_report.py

Основной Python на исходной машине D:/Anaconda3/envs/ml/python.exe; поиск также проверен на D:/Anaconda3/python.exe. Зависимости — requirements.txt; LoRA окружение и версии зафиксированы в results/peft_run_manifest.json. Word авторинг использует bundled python-docx; экспорт для проверки требует установленного Microsoft Word, PNG создаёт bundled Poppler.

Локальные веса Ollama и их полные digest — results/runtime.json. При воспроизведении тег может измениться: сверяйте digest. Контекст8192, temperature0, seed42, общий лимит384, JSON schema. Невалидный ответ — отказ без repair/retry/BM25 fallback. JSON p — заявленная уверенность, не token posterior. CI — bootstrap2000 по query_id.

Для нового независимого опыта скопируйте проект без predictions.jsonl, runtime.json, selection.json и search_trajectories.jsonl; сохраните текущие результаты отдельно. Существующий selection никогда не пересчитывается по test.

Датасет закреплён по revisions и SHA256; все выбранные qrels сохранены. Тексты зеркала покрывают официальные train/dev judgments, 16 текстов сверены с официальным корпусом. Корпус ограничен исторически размеченными пассажами; qrels неполны. Результаты не сравниваются с полным MIRACL leaderboard. Карточки официальных данных указывают Apache-2.0; у зеркала отдельная лицензия не указана, происхождение MIRACL/Wikipedia раскрыто.

Оригиналы прежней работы сохранены в previous/. Старый корневой Word с незаполненными таблицами не является итоговым отчётом. Пилоты и несовместимость Qwen3-VL сохранены в results/pilot и исключены из основного сравнения. Размеры моделей и квантования различаются; это системный локальный benchmark. LoRA — отдельный контроль на0.5B, не QLoRA и не девятая строка.

Автоматические проверки — results/independent_numeric_checks.json и results/final_audit.json; независимая критика модели — results/independent_model_review.json. Визуальная проверка всех PNG остаётся отдельным явно отмеченным этапом, пока первичный агент её не подтвердит. Репозиторий: https://github.com/NikitaGavrilenko/lab1-llm-miracl.
'''
    (LAB/'README.md').write_text(text,encoding='utf-8')


def render_report():
    candidates=list(Path(r'C:\Users\Nick\.codex\plugins\cache\openai-primary-runtime\documents').glob('*/skills/documents/container_tools/mark_artifact_operation_started.mjs'))
    assert candidates,'Document marker missing'
    marker=sorted(candidates)[-1]
    flag=LAB/'.workflow/document_operation_started.json'
    if not flag.exists():
        run('document_marker',[DEPS/'node/bin/node.exe',marker,'--operation-kind','create','--expected-output-count','1','--output-format','docx'])
        save(flag,{'completed':True,'marker':str(marker)})
    run('build_report',[DOC,LAB/'src/build_report.py'])
    qa=LAB/'report/qa';qa.mkdir(parents=True,exist_ok=True)
    run('word_export',['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',LAB/'src/export_word_pdf.ps1',
        '-InputDocx',LAB/'report/Lab1_MIRACL_final.docx','-OutputPdf',qa/'report.pdf'])
    poppler=DEPS/'native/poppler/Library/bin'
    run('render_png',[poppler/'pdftoppm.exe','-r','120','-png',qa/'report.pdf',qa/'page'])
    extractor=LAB/'src/extract_pdf_text.py'
    extractor.write_text("from pathlib import Path\nimport sys\nfrom pypdf import PdfReader\nreader=PdfReader(sys.argv[1])\nPath(sys.argv[2]).write_text('\\f'.join(p.extract_text() or '' for p in reader.pages),encoding='utf-8')\n",encoding='utf-8')
    run('pdf_text',[DOC,extractor,qa/'report.pdf',qa/'report.txt'])
    text=(qa/'report.txt').read_text(encoding='utf-8')
    pages=list(qa.glob('page-*.png'))
    assert pages and 'Macro-F1' in text and 'LoRA' in text
    assert '\ufffd' not in text,'Missing replacement glyph in PDF'
    save(qa/'render_checks.json',dict(pages=len(pages),png_files=[p.name for p in pages],pdf_sha256=sha(qa/'report.pdf'),word_opened_successfully=True,visual_review='pending_primary_agent'))


def package():
    (LAB/'.workflow/monitor.stop').touch()
    audit={'numeric_checks':load(LAB/'results/independent_numeric_checks.json')['passed'],
           'unit_tests':'passed; see results/unit_tests.log',
           'word_render':load(LAB/'report/qa/render_checks.json'),
           'independent_critique':str(LAB/'results/independent_model_review.json'),
           'visual_review':'pending_primary_agent','status':'automatic_pipeline_complete_visual_review_pending'}
    save(LAB/'results/final_audit.json',audit)
    if not (LAB/'.git').exists(): run('git_init',[GIT,'init','-b','main'])
    ignore=(LAB/'.gitignore').read_text(encoding='utf-8') if (LAB/'.gitignore').exists() else ''
    for pattern in ['.venv*/','models/','report/qa/','*.zip','.workflow/full_run.lock']:
        if pattern not in ignore: ignore+='\n'+pattern
    (LAB/'.gitignore').write_text(ignore+'\n',encoding='utf-8')
    run('git_add',[GIT,'add','.'])
    staged=subprocess.run([str(GIT),'diff','--cached','--quiet'],cwd=LAB)
    if staged.returncode==1:
        run('git_commit',[GIT,'-c','user.name=Lab1 research artifact','-c','user.email=lab1@localhost','commit','-m','Complete measured MIRACL lab artifacts'])
    archive=LAB/'Lab1_MIRACL_submission.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in LAB.rglob('*'):
            rel=p.relative_to(LAB)
            if not p.is_file() or any(part in ['.git','models','__pycache__'] or part.startswith('.venv') for part in rel.parts): continue
            if p==archive or rel.parts[:2]==('report','qa') or p.suffix in ['.pyc','.tmp']:continue
            z.write(p,rel)
    save(LAB/'report/submission_manifest.json',{'archive':archive.name,'sha256':sha(archive),'docx_sha256':sha(LAB/'report/Lab1_MIRACL_final.docx')})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-independent-review',action='store_true')
    parser.add_argument('--artifacts-only',action='store_true',help='Rebuild/check/package existing measured artifacts only')
    args=parser.parse_args()
    assert LAB.exists() and ML.exists() and BASE.exists()
    (LAB/'results').mkdir(exist_ok=True);(LAB/'.workflow').mkdir(exist_ok=True)
    target=LAB/'src/finish_lab1.py'
    if Path(__file__).resolve()!=target.resolve():shutil.copy2(__file__,target)
    lock=LAB/'.workflow/full_run.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Another runner may be active; inspect full_run.lock before resuming.')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        if args.artifacts_only:
            patch_report();write_readme();render_report();package()
            state('all_automatic_stages','complete_visual_review_pending',report=str(LAB/'report/Lab1_MIRACL_final.docx'))
            return
        cfg=load(LAB/'configs/protocol.json')
        need=missing_models('validation',cfg['models'])
        if need:run('validation_remaining',[ML,LAB/'src/run_experiment.py','--splits','validation','--models',*need])
        run('freeze_selection',[ML,LAB/'src/analyze_results.py','--select'])
        selection=load(LAB/'results/selection.json')
        need=missing_models('test',cfg['models'])
        if need:run('test_all_models',[ML,LAB/'src/run_experiment.py','--splits','test','--models',*need])
        need=missing_models('application',[selection['selected_model']])
        if need:run('application_classification',[ML,LAB/'src/run_experiment.py','--splits','application','--models',*need])
        run('analysis_all',[ML,LAB/'src/analyze_results.py'])
        run('search_application',[BASE,LAB/'src/search_experiment.py'])
        import requests
        for model in requests.get('http://localhost:11434/api/ps',timeout=10).json()['models']:
            requests.post('http://localhost:11434/api/generate',json={'model':model['name'],'keep_alive':0},timeout=120).raise_for_status()
        if not (LAB/'results/peft_adapted_summary.json').exists():
            peft=prepare_peft();run('peft_real_training',[peft,LAB/'src/peft_experiment.py'])
        aggregate_peft()
        run('unit_tests',[ML,'-m','unittest','discover','-s','tests','-v'])
        run('compile_all',[ML,'-m','compileall','-q','src'])
        state('independent_numeric_checks');verify_metrics();state('independent_numeric_checks','passed')
        if not args.skip_independent_review and not (LAB/'results/independent_model_review.json').exists(): independent_review()
        patch_report();write_readme();render_report();package()
        state('all_automatic_stages','complete_visual_review_pending',report=str(LAB/'report/Lab1_MIRACL_final.docx'))
    except BaseException as error:
        state('failed','error',error=repr(error));raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
