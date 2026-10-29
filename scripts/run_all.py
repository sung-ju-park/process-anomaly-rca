"""전체 분석을 처음부터 다시 실행한다. (약 2분)"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
steps = [["run_tep_detection.py"], ["run_tep_rca.py", "test"], ["run_tep_signature.py"],
         ["run_secom.py"], ["make_sample_reports.py"]]
for s in steps:
    print(f"\n===== {' '.join(s)} =====", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / s[0]), *s[1:]], check=True, cwd=ROOT)
print("\n완료: reports/ 폴더를 확인하세요.")
