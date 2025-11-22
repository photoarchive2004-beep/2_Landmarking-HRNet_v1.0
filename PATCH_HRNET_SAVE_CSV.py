from pathlib import Path
import numpy as np

p = Path("scripts") / "infer_hrnet.py"
text = p.read_text(encoding="utf-8")

old = """def _save_csv(csv_path: Path, coords: np.ndarray) -> None:
    lines = [f"{float(x):.2f},{float(y):.2f}" for x, y in coords]
    csv_path.write_text("\\n".join(lines), encoding="utf-8")
"""

new = """def _save_csv(csv_path: Path, coords: np.ndarray) -> None:
    \"\"\"Сохраняем ландмарки в формате аннотатора:
    одна строка: x1,y1,x2,y2,...,xN,yN
    \"\"\"
    flat = np.asarray(coords, dtype=float).reshape(-1)
    line = ",".join(f"{v:.2f}" for v in flat)
    csv_path.write_text(line + "\\n", encoding="utf-8")
"""

if old not in text:
    raise SystemExit("Не найден старый _save_csv в infer_hrnet.py")

text = text.replace(old, new)
p.write_text(text, encoding="utf-8")
print("OK: infer_hrnet._save_csv patched")
