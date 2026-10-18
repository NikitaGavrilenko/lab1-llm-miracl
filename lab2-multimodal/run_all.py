"""Run the complete laboratory pipeline, stop on failure, preserve stage logs."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-python', default=sys.executable,
                        help='Python with python-docx installed; default is current Python')
    args = parser.parse_args()
    logs = ROOT / 'results' / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    before = ROOT / 'results' / 'predictions_test.csv'
    old_predictions = before.read_bytes() if before.exists() else None
    stages = ['download', 'prepare', 'run', 'plots', 'verify', 'build_report']
    status = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stages': []}
    start = time.perf_counter()
    for stage in stages:
        interpreter = args.report_python if stage == 'build_report' else sys.executable
        script = ROOT / 'src' / (stage + '.py')
        started = time.perf_counter()
        print(f'RUN {stage}', flush=True)
        with (logs / (stage + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run([interpreter, str(script)], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        row = {'stage': stage, 'exit_code': result.returncode,
               'seconds': time.perf_counter() - started}
        status['stages'].append(row)
        if result.returncode:
            status['status'] = 'failed'
            status['failed_stage'] = stage
            (ROOT/'results'/'pipeline_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
            print((logs/(stage+'.log')).read_text(encoding='utf-8'), flush=True)
            raise SystemExit(result.returncode)
        print(f'OK {stage} {row["seconds"]:.1f}s', flush=True)
    status['status'] = 'completed'
    status['wall_seconds'] = time.perf_counter() - start
    status['test_predictions_byte_identical_to_previous_run'] = (
        old_predictions == before.read_bytes() if old_predictions is not None else None)
    status['summary'] = json.loads((ROOT/'results'/'run_summary.json').read_text(encoding='utf-8'))
    (ROOT/'results'/'pipeline_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
