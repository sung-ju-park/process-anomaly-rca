"""샘플 RCA 리포트 생성 (고장 시작 이후 첫 경보 기준)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from engine import Engine  # noqa: E402
from report import render  # noqa: E402
from tep_data import TEST_ONSET, load  # noqa: E402
from tep_meta import FAULTS  # noqa: E402

E = Engine()
out = ROOT / "reports" / "rca_reports"
out.mkdir(parents=True, exist_ok=True)
for f in (4, 6, 12):
    X = load(f, "test").values
    ta = E.first_alarm(X, TEST_ONSET)
    F = E.investigate(X, ta, f"테스트 시나리오 {f:02d} — {FAULTS[f][0]} ({FAULTS[f][1]})")
    (out / f"rca_report_fault{f:02d}.html").write_text(render(F, X, E.mon.sc), encoding="utf-8")
    print(f, ta, F.causes["변수"].head(3).tolist(), F.similar["과거 사례"].tolist())
