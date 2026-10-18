"""Build the final report only from frozen experimental artifacts."""
import csv
import hashlib
import json
from pathlib import Path
from docx import Document
from docx.enum.section import WD_SECTION_START, WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'report/Lab1_MIRACL_final.docx'
LABELS={'qwen2.5-coder:1.5b':'Coder 1.5B','llama3.2:3b':'Llama 3.2 3B',
'gemma3:4b':'Gemma 3 4B','phi3.5:3.8b':'Phi 3.5 3.8B','qwen2.5-coder:14b':'Coder 14B',
'qwen3:14b':'Qwen3 14B','qwen3:30b':'Qwen3 30B MoE','qwen2.5-coder:32b':'Coder 32B'}


def load(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def fmt(x,n=3):
    return 'неприм.' if x is None else f'{x:.{n}f}'.replace('.',',')


def ci(x):
    return '['+'; '.join(fmt(v) for v in x)+']'


def configure(sec,landscape=False):
    sec.orientation=WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
    sec.page_width=Cm(29.7 if landscape else 21)
    sec.page_height=Cm(21 if landscape else 29.7)
    sec.top_margin=sec.bottom_margin=Cm(1.8)
    sec.left_margin=sec.right_margin=Cm(1.8)
    sec.header_distance=sec.footer_distance=Cm(.8)


def build():
    summary=load('results/summary.json'); selected=load('results/selection.json')
    passport=load('data/prepared/passport.json'); lit=load('report/literature.json')
    runtime=load('results/runtime.json'); config=load('configs/protocol.json')
    sources=load('configs/model_sources.json')
    groups=summary['groups']; test=[g for g in groups if g['split']=='test']
    assert len(test)==16 and all(g['query_count']==40 and g['document_count']==416 for g in test)
    sm=selected['selected_model']; mode=selected['selected_mode']
    winner=next(g for g in test if g['model']==sm and g['mode']==mode)
    doc=Document();configure(doc.sections[0])
    for name,size in [('Normal',12),('Title',18),('Heading 1',14),('Heading 2',12),('Caption',10)]:
        style=doc.styles[name];style.font.name='Times New Roman';style.font.size=Pt(size)
        style.font.color.rgb=RGBColor(0,0,0)
        style._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Times New Roman')
        style.paragraph_format.space_after=Pt(6)
        style.paragraph_format.line_spacing=1.15
    for name in ['Heading 1','Heading 2']:
        doc.styles[name].font.bold=True
        doc.styles[name].paragraph_format.keep_with_next=True
    doc.styles['Normal'].paragraph_format.widow_control=True
    foot=doc.sections[0].footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)
    table_log=[]
    def p(text,style=None):
        r=doc.add_paragraph(str(text),style);return r
    def h(text): return doc.add_heading(text,level=1)
    def sub(text): return doc.add_heading(text,level=2)
    def table(headers,rows,widths=None,size=9.5):
        t=doc.add_table(rows=1,cols=len(headers));t.style='Table Grid';t.autofit=False
        for i,name in enumerate(headers):t.rows[0].cells[i].text=str(name)
        pr=t.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');pr.append(repeat)
        for vals in rows:
            cells=t.add_row().cells
            for i,value in enumerate(vals): cells[i].text=str(value)
        if widths:
            for row in t.rows:
                for i,width in enumerate(widths):row.cells[i].width=Cm(width)
        for ri,row in enumerate(t.rows):
            props=row._tr.get_or_add_trPr();keep=OxmlElement('w:cantSplit');props.append(keep)
            for cell in row.cells:
                for par in cell.paragraphs:
                    par.paragraph_format.space_after=Pt(3);par.paragraph_format.space_before=Pt(2)
                    par.paragraph_format.line_spacing=1
                    for run in par.runs:run.font.name='Times New Roman';run.font.size=Pt(size);run.bold=ri==0
                if ri==0:
                    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'EDEDED');cell._tc.get_or_add_tcPr().append(shade)
        table_log.append({'headers':headers,'rows':rows});p('')
        return t
    def fig(path,caption):
        path=ROOT/path
        assert path.exists(),path
        doc.add_picture(str(path),width=Cm(17.1));p(caption,'Caption')
    p('Финансовый университет при Правительстве Российской Федерации').alignment=WD_ALIGN_PARAGRAPH.CENTER
    p('Прикладные задачи машинного обучения').alignment=WD_ALIGN_PARAGRAPH.CENTER
    p('Лабораторная работа 1','Title').alignment=WD_ALIGN_PARAGRAPH.CENTER
    p('Сравнение языковых моделей на русскоязычном MIRACL','Title').alignment=WD_ALIGN_PARAGRAPH.CENTER
    p('Оценка релевантности и последовательное взаимодействие с поиском').alignment=WD_ALIGN_PARAGRAPH.CENTER
    p('Группа ПМ23-2').alignment=WD_ALIGN_PARAGRAPH.CENTER
    p('Москва 2026').alignment=WD_ALIGN_PARAGRAPH.CENTER
    p('Тема ВКР: «Разработка языковой модели для многошагового взаимодействия с поисковыми инструментами агентной системы».')
    p(f'Проведено сравнение восьми локальных LLM в zero-shot и few-shot на общем наборе из 40 запросов и 416 оценок релевантности. По правилу, зафиксированному до test, выбрана {LABELS[sm]} в режиме {mode}. На test её Macro-F1 равен {fmt(winner["macro_f1_operation"])} с 95% ДИ {ci(winner["macro_f1_operation_ci95"])}. Отдельно выполнены поиск по корпусу из 41 007 пассажей и реальная LoRA-адаптация малой модели.')
    p('Все таблицы формируются из сохранённых ответов моделей. Числа из демонстрационного примера методички в результаты не переносились. Дата эксперимента 18 октября 2026 года.')
    doc.add_page_break()
    h('1 Постановка исследования')
    p('Цель работы — выбрать доступную языковую модель для компонента, который оценивает найденные свидетельства и взаимодействует с поиском. Для ВКР важно отделять качество релевантности от способности строить многошаговую поисковую траекторию.')
    p(lit['research_gap']['research_question'])
    p('Основной эксперимент — бинарная оценка релевантности: каждому запросу предъявляется весь его размеченный пул пассажей без меток. Одна генерация возвращает список оценок для всех кандидатов. Это контекстная классификация и условный reranking; модель не извлекает документы из корпуса на этом этапе.')
    p('Отдельный прикладной эксперимент использует настоящий фиксированный индекс из 41 007 пассажей. Следующий поисковый запрос зависит от реальной предыдущей выдачи, а политика может остановиться. Это ограниченный инструментальный поиск с обратной связью. MIRACL не содержит эталонных агентных траекторий и не доказывает качество multi-hop reasoning.')
    for i,hyp in enumerate(lit['research_gap']['hypotheses'],1):p(f'H{i}. {hyp}')
    p('Аудит предыдущей работы выявил незаполненные результаты, отсутствие реальных запусков, слабый паспорт данных, синтетические few-shot примеры и BM25 fallback в метриках LLM. Старые материалы сохранены в previous. Итоговая работа использует реальные демонстрации из train, самостоятельные отказы и отдельный поисковый опыт.')
    h('2 Обзор литературы')
    sub('2.1 Поиск и отбор источников')
    p('Поисковый запрос: ("large language model" OR LLM) AND (retrieval OR reranking OR "search agent" OR "multi-step") AND (MIRACL OR Russian OR multilingual). Для адаптации отдельно использовались LoRA и QLoRA. Проверялись ACL Anthology, arXiv, OpenReview, proceedings и авторские репозитории.')
    p('PRISMA-подобный журнал описывает целенаправленный отбор, а не исчерпывающий систематический обзор. Рассмотрено 20 явно перечисленных кандидатов; включено 15, исключено 5. Объём всей поисковой выдачи и число её дублей не подсчитывались. Для ключевых работ сверены релевантные разделы протокола и метрик; для BEIR и RuBQ 2.0 глубина чтения ограничена первичными описаниями и авторским кодом.')
    table(['Этап','Число','Основание'],[['Рассмотренные кандидаты','20','Оригинальные работы по retrieval, агентам, PEFT и русским данным'],['Включённые работы','15','Задают данные, методы, протокол или ограничения лабораторной'],['Исключённые работы','5','Дублирующая версия RuBQ, вторичный обзор и менее близкие методы']],[4,2,11.3])
    sub('2.2 Сопоставление подходов')
    for text in lit['narrative']:p(text)
    p(lit['research_gap']['statement'])
    p('Практический вывод обзора: качество поиска следует оценивать отдельно от генерации, сравнение вести на одинаковых данных, а обучение политики требует самостоятельного опыта с траекториями. Публикационные результаты разных датасетов не используются как ожидаемые показатели лабораторной.')
    sec=doc.add_section(WD_SECTION_START.NEW_PAGE);configure(sec,True)
    sub('2.3 Матрица пятнадцати работ')
    p('Ниже объединены задача и архитектура, данные и протокол, метрики, ограничения и идея репликации. Полная машиночитаемая матрица с авторскими ссылками и кодом находится в report/literature_matrix.csv. Номера соответствуют списку источников.')
    rows=[]
    for x in lit['papers']:
        rows.append([f'[{x["id"]}] {x["key"]}\n{x["year"]}',x['task']+' '+x['architecture'],x['data']+' '+x['protocol'],'; '.join(x['metrics']),x['limitation']+'\nРепликация: '+x['replication']])
    table(['Работа','Задача и архитектура','Данные и протокол','Метрики','Ограничение и репликация'],rows,[2.3,5.5,6.1,3.2,9],9)
    sec=doc.add_section(WD_SECTION_START.NEW_PAGE);configure(sec)
    h('3 Данные и подготовка')
    p('Источник запросов и меток — официальный MIRACL ru. Метка 1 означает релевантный пассаж, 0 — оценённый нерелевантный пассаж; отсутствие оценки не означает метку 0. Исторические negative passages уже размечены людьми, случайные отрицательные примеры не подмешиваются.')
    p('Официальные topics/qrels закреплены на revision '+passport['official_topics_qrels_revision']+'. Тексты получены из компактного зеркала sohey1024/miracl-ru-corpus, revision '+passport['text_mirror_revision']+'. Его 41 007 ID в точности совпадают с объединением официальных ru train/dev qrels. Проверены все ID и метки; для 16 пересекающихся пассажей из 400 прочитанных официальных строк заголовки и тексты совпали полностью.')
    p('Карточки MIRACL и корпуса указывают Apache-2.0. У зеркала отдельная лицензия в карточке не указана; происхождение текстов — MIRACL/Wikipedia. Сохранены ссылки на первоисточник, ревизии и SHA256. Это ограничение происхождения зеркала, а не основание приписывать ему новую лицензию.')
    table(['Часть','Источник','Запросы','Метки 1','Метки 0','Всего'],[[k,('official train' if k in ['train','validation'] else 'official dev'),v['queries'],v['positive_judgments'],v['negative_judgments'],v['judgments']] for k,v in passport['splits'].items()],[2.4,3.7,2.2,2.2,2.2,2.6])
    p('Train используется для демонстраций и LoRA; validation — для выбора модели и режима. Test открывается после фиксации выбора. Application — отдельные 20 запросов для переноса выбранного решения. Числовые query_id разных официальных split пересекаются, поэтому в коде используются пространства имён train: и dev:.')
    p('Выборка получена после сортировки ID и перемешивания с seed 42 для official train и 43 для official dev. Сохраняются первые подходящие запросы с обоими классами и не более 20 judgments. В рассмотренной последовательности исключены 51 запрос, не удовлетворяющий условиям, и 3 запроса с межвыборочным пересечением документа. Отбор не зависит от успеха моделей или сложности BM25.')
    p('Для утечек проверяются docid, нормализованные тексты, близкие тексты по Jaccard множеств токенных 3-шинглов при пороге 0,85 и близкие формулировки запросов. Между четырьмя использованными частями пересечений не найдено. Проверка охватывает выбранные данные, но не может исключить попадание Wikipedia или MIRACL в предобучение самих LLM.')
    p(f'Во всём корпусе найдено {passport["corpus"]["normalized_duplicate_text_groups"]} групп точных нормализованных дублей, {passport["corpus"]["extra_duplicate_text_passages"]} повторных пассажей. Они сохраняют официальные ID; объединение нарушило бы qrels. Пустых текстов {passport["missing_values"]["empty_text"]}, пустых заголовков {passport["missing_values"]["empty_title"]}. Медиана длины пассажей корпуса — {passport["corpus"]["characters"]["median"]} символа. Временные метки наблюдений не предоставлены; разбиение групповое случайное, а не временное.')
    p('Изоляция выполнена по query_id, docid и текстам, а не по статьям Wikipedia целиком. Независимый аудит обнаружил одну общую статью между train и test при разных непохожих пассажах. Утечки одинаковой пары нет, но полностью независимый перенос на новые статьи этим разбиением не проверяется.')
    p('В сохранённых данных тексты полные. Модель видит не более 900 символов пассажа; общий бюджет текстов — 14 000 символов. Заголовки ограничены 120 символами. У всех моделей одинаковые фрагменты и один детерминированно перемешанный порядок. Усечение может удалить ответ из конца документа; анализ ошибок отдельно учитывает исходную длину.')
    h('4 Единый протокол сравнения')
    sub('4.1 Модели и ресурсы')
    modelrows=[]
    for m in config['models']:
        d=runtime['models'][m]['details']
        modelrows.append([m,d['parameter_size'],d['quantization_level'],sources[m]['license'],runtime['models'][m].get('preflight_prompt_tokens')])
    table(['Ollama ID','Размер','Квантование','Лицензия','Preflight tokens'],modelrows,[4.5,2,2.5,4.1,2.2])
    p('Измерения выполнены последовательно на NVIDIA RTX 3090 24 ГБ, Ryzen 7 5700X, Windows 10 и Ollama 0.40.4. Основной Python 3.10.20. Сравниваются конкретные локальные сборки и размеры, а не восемь независимых семейств. Большая доля Qwen обусловлена уже установленными весами. Различия квантования входят в ограничения системного сравнения; контрольная переквантизация всех моделей не выполнялась.')
    p('Проверка совместимости локальной Qwen3-VL показала продолжение рассуждения при think=false и исчерпание 384 токенов до финального JSON. Десять попыток сохранены отдельно; до открытия test модель заменена на Phi-3.5. Алиас claude-local с теми же весами, что Qwen2.5-Coder-14B, отдельной моделью не считается.')
    sub('4.2 Генерация и контроль формата')
    p('Для каждой модели выполнены zero-shot и two-shot с одинаковыми двумя размеченными примерами из train: один положительный, один отрицательный. Демонстрационные оценки 0,9 и 0,1 — фиксированные якоря формата и направления уверенности, а не эталонные вероятности. Модель получает все фрагменты одного запроса одновременно; их оценки могут зависеть друг от друга.')
    p('Параметры: temperature 0, seed 42, контекст 8192, max_new_tokens 384, top_k 1, top_p 1, repeat_penalty 1. Thinking отключён там, где он поддерживается. JSON-схема задаёт ровно одно число [0;1] на каждый фрагмент. Тем самым доля невалидных ответов измеряется в условиях ограниченного декодирования; она не характеризует свободное следование JSON-инструкции.')
    p('Перед измерением каждой модели выполнен прогрев на самом длинном по тексту few-shot примере из train и проверено число входных токенов. Для каждого запроса сохраняются полный prompt, хеш, сырой ответ, счётчики токенов, latency и digest весов. Ключ кэша включает prompt, данные, параметры, модель и версию сервера. Автоматические исправления ответа и повторные попытки не используются.')
    p('Невалидный ответ означает отказ сразу по всему запросу. Для эксплуатационного Macro-F1 отказ считается отдельной предсказанной меткой 2 и штрафует обе целевые категории. MCC, Brier и ECE считаются только по валидным ответам вместе с покрытием. В reranking отказ даёт пустую выдачу и нулевую метрику. BM25 fallback не добавляется к результату LLM.')
    sub('4.3 Метрики и статистика')
    p('Основной показатель — Macro-F1 двух классов при пороге p ≥ 0,5. Дополнительно измеряются MCC, precision/recall/F1 релевантного класса, Brier и ECE с 10 фиксированными интервалами. Brier — среднее (p − y)²; ECE сопоставляет среднюю заявленную оценку и частоту релевантности. Это калибровка сообщаемой моделью уверенности, а не оценка её token posterior.')
    p('Вспомогательные nDCG@10, MRR@10 и Recall@3/10 считаются после сортировки этих оценок внутри judged pool. Ничьи разрешаются фиксированным перемешанным порядком. В test у большинства запросов ровно 10 кандидатов, поэтому Recall@10 близок к насыщению и не доказывает полноту поиска. В глобальном поисковом опыте Recall@10 имеет другое, содержательное значение.')
    p('95% интервалы построены percentile bootstrap из 2000 повторов по целым query_id, seed 42. Пары zero/few сравниваются на одинаковых запросах. Интервалы не корректируются за множество сравнений и не дают основания объявлять статистического победителя по одному пересечению CI. Latency — end-to-end время тёплого запроса с подготовленным prompt; загрузка и прогрев исключены и сохранены отдельно. Память — resident size_vram Ollama после прогрева; журнал GPU дополнительно записывается каждые две секунды, это не непрерывный аппаратный пик.')
    h('5 Результаты на общем тесте')
    p('Каждая строка относится к тем же 40 запросам и 416 judgments. Режим Z — zero-shot, F — few-shot. Интервалы относятся к эксплуатационному Macro-F1; Brier и ECE — к валидным ответам.')
    rows=[]
    for m in config['models']:
        for mo in ['zero-shot','few-shot']:
            g=next(g for g in test if g['model']==m and g['mode']==mo)
            rows.append([LABELS[m], 'Z' if mo=='zero-shot' else 'F',fmt(g['macro_f1_operation'])+' '+ci(g['macro_f1_operation_ci95']),fmt(g['mcc_valid']),fmt(g['brier_valid']),fmt(g['ece10_valid']),fmt(100*g['invalid_query_rate'],1)])
    table(['Модель','Реж.','Macro-F1 и 95% ДИ','MCC','Brier','ECE','Отказ %'],rows,[3,1.4,4.2,1.7,1.7,1.7,1.6],9)
    baselines=[g for g in summary['baselines'] if g['split']=='test' and 'macro_f1_operation' in g]
    table(['Baseline','Macro-F1','Brier','ECE'],[[g['model'],fmt(g['macro_f1_operation']),fmt(g['brier_valid']),fmt(g['ece10_valid'])] for g in baselines],[7,3.4,3.4,3.4])
    fig('figures/comparison_test.png','Рисунок 1 Macro-F1 восьми моделей и 95% интервалы по query_id')
    table(['Модель','Реж.','p50 с','p95 с','Resident VRAM ГиБ','tokens/s'],[[LABELS[g['model']],'Z' if g['mode']=='zero-shot' else 'F',fmt(g['latency_p50_s'],2),fmt(g['latency_p95_s'],2),fmt(g['resident_vram_gib'],2),fmt(g['generation_tokens_per_s_median'],1)] for m in config['models'] for g in sorted([x for x in test if x['model']==m],key=lambda x:x['mode'],reverse=True)],[3.3,1.2,2.5,2.5,4.5,2.8])
    fig('figures/calibration_test.png','Рисунок 2 Калибровка заявленной уверенности моделей на test')
    # Finished artifact tables
    paired=load('results/paired_comparisons.json')
    pairrows=[r for r in paired if r['split']=='test' and r['metric']=='macro_f1_operation']
    table(['Модель','Few минус zero','95% парный ДИ'], [[LABELS[r['model']],fmt(r['delta']),ci([r['ci95_low'],r['ci95_high']])] for r in pairrows],[5.3,4,8])
    improved=sum(r['delta']>0 for r in pairrows)
    p(f'Few-shot имеет более высокий точечный Macro-F1 у {improved} из 8 моделей. Поэтому универсальное улучшение от двух демонстраций не предполагается. Каждая разность оценивается на парных query_id; отсутствие коррекции множественных сравнений ограничивает выводы.')
    table(['Модель','Реж.','nDCG@10','MRR@10','Recall@3','Recall@10'],[[LABELS[g['model']],'Z' if g['mode']=='zero-shot' else 'F',fmt(g['ndcg10_judged']),fmt(g['mrr10_judged']),fmt(g['recall3_judged']),fmt(g['recall10_judged'])] for g in test],[3.4,1.2,3.1,3.1,3.1,3.4])

    # Table schemas are emitted verbatim to retain every measured field in the artifact bundle.
    p('Парные разности few-shot минус zero-shot и интервалы приведены в results/paired_comparisons.csv. Few-shot не считается улучшением только по росту точечной оценки: учитываются ширина интервала и изменение задержки.')
    h('6 Выбор модели и анализ ошибок')
    p('Выбор выполнен только на 16 validation-запросах до запуска test. Среди пар модель/режим с invalid ≤ 5% максимизируется U = 0,55 Macro-F1 + 0,15 (MCC+1)/2 + 0,15 (1−Brier) + 0,10 (1−p95/max p95) + 0,05 (1−VRAM/max VRAM). Максимумы для нормировки взяты только по validation. При ничьей применяется детерминированный порядок в коде; порог 0,5 не настраивается по test.')
    p(f'Зафиксирован выбор {LABELS[sm]} в режиме {mode}. На test: Macro-F1 {fmt(winner["macro_f1_operation"])}, MCC {fmt(winner["mcc_valid"])}, Brier {fmt(winner["brier_valid"])}, p95 {fmt(winner["latency_p95_s"],2)} с. Выбранная пара переносится без замены по итогам test. Это решение для локального прототипа, а не универсальный рейтинг LLM.')
    table(['Кандидат validation','Macro-F1','Brier','p95 с','Utility'],[[LABELS[r['model']]+' '+r['mode'],fmt(r['macro_f1_operation']),fmt(r['brier_valid']),fmt(r['latency_p95_s'],2),fmt(r['utility'])] for r in sorted(selected['candidates'],key=lambda r:r['utility'] if r['utility'] is not None else -1,reverse=True)],[6,2.8,2.8,2.8,2.9])
    robust=load('results/robustness.json')
    table(['Модель','Смена класса %','Среднее |Δp|','Δ Macro-F1'],[[LABELS[r['model']],fmt(100*r['flip_rate_valid'],1),fmt(r['mean_absolute_probability_delta']),fmt(r['macro_f1_delta'])] for r in robust if r['split']=='test'],[5.2,4,4,4.1])

    p('Для первых восьми запросов каждой evaluation-части у всех моделей повторён zero-shot с альтернативной формулировкой смысловой инструкции. Данные, порядок, формат и бюджет сохранены; проверяются доля сменившихся классов и изменение качества. Полные результаты находятся в results/robustness.csv. Проверка касается формулировки инструкции, а не независимого многоязычного переноса или произвольной атаки на prompt.')
    errors=load('results/error_examples.json')
    selected_errors=[e for e in errors if e.get('model')==sm and e.get('mode')==mode and e.get('split')=='test']
    for e in selected_errors[:3]:
        p('Запрос '+e['qid']+': '+e['query'])
        p('Фрагмент «'+e['title']+'»: '+e['text'])
        p(f'Эталон {e["true_label"]}, прогноз {e["predicted_label"]}, p={fmt(e["p_relevant"])}. Текст ошибки взят из реально показанного модели входа.')
    slices=[r for r in load('results/error_slices.json') if r['model']==sm and r['mode']==mode and r['split']=='test']
    table(['Разрез','Значение','Пар','Покрытие','Доля ошибок','Brier'],[[r['slice_type'],r['slice_value'],r['documents'],fmt(r['coverage']),fmt(r['error_rate_operation']),fmt(r['brier_valid'])] for r in slices],[3.5,3.6,1.7,2.5,3.1,2.9])
    app=next(g for g in groups if g['split']=='application' and g['model']==sm and g['mode']==mode)
    p(f'На отдельном срезе application из 20 запросов классификатор получает Macro-F1 {fmt(app["macro_f1_operation"])} с 95% ДИ {ci(app["macro_f1_operation_ci95"])}, MCC {fmt(app["mcc_valid"])} и Brier {fmt(app["brier_valid"])}. Пара модель/режим не менялась после validation.')

    p('В dev:1711 модель отвергла пассаж о районе Тольятти, хотя в нём прямо названа Волга. В dev:2212 дата рождения нужного Хэмилтона получила 0, а год рождения другого человека по имени Льюис — 1. Это наблюдаемые ошибки связи сущности и факта. Их точную внутреннюю причину без контрфактических проверок установить нельзя.')
    p('Первые два ошибочно отвергнутых текста содержат 298 и 303 символа и не усечены. Следовательно, эти ошибки нельзя объяснить лимитом 900 символов. Крайние оценки 0 и 1 при неверном решении показывают риск чрезмерной заявленной уверенности.')
    p('Разрезы по истинному классу, исходной длине текста и типу вопроса сохранены в results/error_slices.csv. Для длинных пассажей усечение — самостоятельная угроза валидности. Весь набор русский, поэтому сравнение языков по нему невозможно. Пункт методички о типах финансового сообщения адаптирован к области ВКР: вместо финансовой тональности используется релевантность, вместо жанра финансовой новости — тип вопроса Wikipedia.')
    h('7 Применение выбранной модели к поиску')
    p('На application используются 20 новых запросов и фиксированный корпус 41 007 пассажей. В индекс входят исторически размеченные пассажи train/dev, но на каждом шаге модель видит только реальную выдачу BM25, без меток. Положительные документы не вставляются по query_id. Корпус построен из исторического judged pool, поэтому опыт легче полнокорпусного MIRACL и не сопоставим с его leaderboard.')
    p('Исходный запрос выполняется BM25. Модель наблюдает пять коротких результатов и выбирает search с новым запросом либо stop. Максимум — три обращения к индексу, два решения LLM. Повтор запроса, ошибка формата или исчерпание бюджета завершают цикл. Финальный список объединяется RRF с k=60. Сравниваются исходный BM25 и детерминированное расширение по заголовкам с тем же максимальным бюджетом.')
    search=load('results/search_summary.json')
    names={'original_bm25':'BM25','lexical_expansion':'Расширение заголовками','winner_feedback':'LLM с обратной связью'}
    table(['Метод','nDCG@10 и ДИ','Recall@10','MRR','Поисков','LLM','Время с'],[[names[k],fmt(v['ndcg10']['mean'])+' '+ci(v['ndcg10']['ci95']),fmt(v['recall10']['mean']),fmt(v['mrr']['mean']),fmt(v['search_calls']['mean'],2),fmt(v['llm_calls']['mean'],2),fmt(v['latency_s']['mean'],4)] for k,v in search['methods'].items()],[3.5,4.1,2.2,1.8,1.8,1.5,2.4])
    fig('figures/search_comparison.png','Рисунок 3 Поиск на двадцати новых запросах и 95% интервалы')
    for comparator,diffs in search['feedback_minus_comparator'].items():
        d=diffs['ndcg10']
        p(f'По сравнению с {names[comparator]} разность nDCG@10 равна {fmt(d["mean_difference"])} с парным ДИ {ci(d["paired_ci95"])}; улучшений {d["wins"]}, совпадений {d["ties"]}, ухудшений {d["losses"]} из 20.')
    coverage=search['methods']['winner_feedback']['judged_coverage10']['mean']
    p(f'Доля top-10 с известной оценкой равна {fmt(coverage)}. При подсчёте TREC-метрик неразмеченные документы получают нулевой gain; это техническое соглашение при неполных qrels, а не доказательство их нерелевантности.')
    trajectories=[json.loads(s) for s in (ROOT/'results/search_trajectories.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    for tr in sorted(trajectories,key=lambda t:len(t['methods']['winner_feedback']['searches']),reverse=True)[:1]:
        fb=tr['methods']['winner_feedback']
        p('Пример многошаговой траектории '+tr['qid']+': '+tr['query']+' Поисковые запросы: '+' → '.join(s['query'] for s in fb['searches'])+'. Завершение: '+fb['status']+'.')

    p('При интеграции в ВКР сохраняются search → observation → action → stop, версия индекса и бюджет действий. Прототип должен передавать сомнительные случаи на проверку при невалидном ответе, повторении запроса, отсутствии свидетельств или исчерпании бюджета. Поиск не выдаётся за доказанно правильный финальный ответ: MIRACL не содержит необходимых answer labels.')
    h('8 Параметрически эффективная адаптация')
    peft=load('results/peft_summary.json')
    p('Дополнительный контроль — Qwen2.5-0.5B-Instruct до и после настоящей LoRA-адаптации. Он не подменяет восьмимодельный опыт: используются отдельная малая модель, другой inference engine и token-aware вход до 256 токенов. Обучение: 128 train-пар, по одной положительной и отрицательной на запрос; rank 8 на q_proj/v_proj, AdamW, learning rate 10⁻⁴, одна эпоха, не более 64 шагов, seed 42.')
    p('Цель — бинарная cross-entropy по логитам следующих токенов 0 и 1. Вероятность определяется softmax только над этими двумя токенами. Это другая шкала, чем сообщаемая JSON-уверенность основного опыта. Test не используется для настройки. Балансировка train меняет априорную частоту класса относительно естественного test; Brier может ухудшиться. Выполнена LoRA, не QLoRA: квантованное обучение не имитируется названием.')
    table(['Часть','Состояние','Macro-F1 и ДИ','MCC','Brier','ECE'],[[sp,stage,fmt(peft[stage]['metrics'][sp]['macro_f1'])+' '+ci(peft[stage]['metrics'][sp]['ci95']['macro_f1']),fmt(peft[stage]['metrics'][sp]['mcc']),fmt(peft[stage]['metrics'][sp]['brier']),fmt(peft[stage]['metrics'][sp]['ece'])] for sp in ['validation','test','application'] for stage in ['base','adapted']],[2.5,2.1,5.1,2.2,2.2,3.2])
    p(f'В LoRA-контроле при лимите 256 токенов усечено {fmt(100*peft["base"]["metrics"]["test"]["truncated_fraction"],1)}% тестовых пассажей. Инструкция и вопрос сохраняются, но модель часто видит лишь начало фрагмента. Это существенное ограничение переноса и дополнительная причина не сравнивать его напрямую с основным опытом.')
    training=peft['training']; manifest=peft['manifest']
    p(f'Выполнено {training["steps"]} шагов; обучаемых параметров {training["trainable_parameters"]:,} из {training["total_parameters"]:,}. Обучение заняло {fmt(training["elapsed_s"],1)} с на {manifest["device"]}. Наблюдаемый максимум RSS процесса {fmt(training["rss_observed_peak_bytes"]/2**30,2)} ГиБ; это оперативная память, а не VRAM.')
    before=peft['base']['metrics']['test']['macro_f1'];after=peft['adapted']['metrics']['test']['macro_f1']
    p(f'Изменение тестового Macro-F1 после LoRA равно {fmt(after-before)}. Это фактический результат короткой адаптации; по одной малой модели и одной эпохе нельзя делать вывод о всех способах PEFT.')

    h('9 Ограничения и связь с ВКР')
    for text in [
    'Сорок тестовых запросов дают широкие интервалы. Отбор запросов с обоими классами и строгая проверка утечек меняют исходное распределение; на запросы без положительных документов результат не распространяется.',
    'Основная классификация получает размеченный пул и не измеряет candidate recall. Почти насыщенный Recall@10 нельзя трактовать как успех retrieval. В поисковом опыте ограниченный корпус содержит все исторически размеченные документы, но qrels для других документов неполны.',
    'Один детерминированный порядок кандидатов устраняет разницу между моделями, но не оценивает устойчивость к перестановкам. Few-shot и усечение могут смещать оценки. Возможна неизвестная утечка MIRACL/Wikipedia в предобучение.',
    'Измеряется конкретная локальная сборка с её шаблоном и квантованием. Модели не охватывают всё разнообразие LLM. Задержка одного запроса включает серверные накладные расходы; для многопользовательской нагрузки нужен отдельный throughput benchmark.',
    'Последовательный поиск с обратной связью не равен multi-hop QA. Для ВКР следующий опыт должен использовать ответы и несколько необходимых свидетельств, отдельные teacher trajectories и оценку политики до и после SFT/QLoRA.']:
        p(text)
    best_test=max(test,key=lambda g:g['macro_f1_operation'])
    p(f'Максимальный точечный Macro-F1 на test получен у {LABELS[best_test["model"]]} {best_test["mode"]}: {fmt(best_test["macro_f1_operation"])}. Зафиксированная по validation пара имеет {fmt(winner["macro_f1_operation"])}. Её p95 {fmt(winner["latency_p95_s"],2)} с против {fmt(best_test["latency_p95_s"],2)} с, resident VRAM {fmt(winner["resident_vram_gib"],2)} против {fmt(best_test["resident_vram_gib"],2)} ГиБ. Максимум качества и практический выбор по нескольким критериям здесь различаются; test не используется для замены выбранного режима.')
    hyp=[r for r in load('results/paired_comparisons.json') if r['split']=='test' and r['metric']=='macro_f1_operation']
    better=sum(r['delta']>0 for r in hyp);worse=sum(r['delta']<0 for r in hyp)
    p(f'В проверке H1 few-shot улучшил точечный Macro-F1 у {better} из 8 моделей, ухудшил у {worse}, не изменил у {8-better-worse}. Для данной пары демонстраций гипотеза об общем преимуществе few-shot не подтвердилась. Интервалы большинства различий включают ноль; перенос вывода на другие демонстрации не проверен.')
    h('10 Выводы и воспроизведение')
    p(f'Основной результат — воспроизводимое сравнение восьми локальных моделей в двух режимах и выбор {LABELS[sm]} по заранее заданному многокритериальному правилу. Результаты классификации, условного reranking и инструментального поиска представлены раздельно. Обзор пятнадцати работ обосновывает дальнейшее обучение именно поисковой политики на проверенных траекториях.')
    p('Источник всех таблиц — results/predictions.jsonl, summary.json, search- и peft-артефакты. Данные и хеши находятся в data/prepared/passport.json, параметры — в configs, воспроизведение и проверки — в README.md. Матрица литературы содержит реальные научные источники и ссылки на код.')
    p('Локальный репозиторий студента: D:\\Projects\\University_Current\\lab1-llm-miracl. Копия проекта со скриптами, конфигурациями, данными и результатами прилагается к отчёту. Публичная ссылка не выдумывается: удалённый репозиторий не был настроен.')
    h('Список источников')
    for x in lit['papers']:
        p(f'[{x["id"]}] {x["authors"]}. {x["title"]}. {x["venue"]}, {x["year"]}. '+x['url']+' Код: '+x['code_url'])
    p('Данные MIRACL: https://huggingface.co/datasets/miracl/miracl ; корпус: https://huggingface.co/datasets/miracl/miracl-corpus ; использованное зеркало: https://huggingface.co/datasets/sohey1024/miracl-ru-corpus . Дата проверки источников 18.10.2026.')
    for m,info in sources.items():p(m+': '+info['official_model_card'])
    doc.core_properties.title='Лабораторная работа 1 Сравнение LLM на MIRACL ru'
    doc.core_properties.subject='Прикладные задачи машинного обучения'
    doc.core_properties.author='Студент группы ПМ23-2'
    # Remove theme font overrides and the template's decorative title border.
    for style in doc.styles:
        for fonts in style._element.iter(qn('w:rFonts')):
            for attr in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
                fonts.attrib.pop(qn('w:'+attr),None)
        for border in list(style._element.iter(qn('w:pBdr'))):
            border.getparent().remove(border)
    for fonts in doc._element.iter(qn('w:rFonts')):
        for attr in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
            fonts.attrib.pop(qn('w:'+attr),None)
    for border in list(doc._element.iter(qn('w:pBdr'))):
        border.getparent().remove(border)
    for t in doc.tables:
        if len(t.rows)<=5:
            for row in t.rows[:-1]:
                for cell in row.cells:
                    for par in cell.paragraphs:par.paragraph_format.keep_with_next=True
    doc.save(OUT)
    (ROOT/'report/tables_export.json').write_text(json.dumps(table_log,ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT)


if __name__=='__main__': build()