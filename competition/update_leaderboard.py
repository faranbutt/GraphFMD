import csv
from pathlib import Path
from datetime import datetime
import sys
try:
    team, run_id, score, author_type, model, notes = sys.argv[1:7]
except ValueError:
    print("Error: Missing arguments.")
    sys.exit(1)

CSV_PATH = Path("leaderboard/leaderboard.csv")
CSV_PATH.parent.mkdir(exist_ok=True)
headers = ["team", "run_id", "macro_f1", "submitted_at", "author_type", "model", "notes"]
if CSV_PATH.exists():
    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["team"] == team:
                print(f"Submission rejected: Team '{team}' has already submitted.")
                sys.exit(0)

exists = CSV_PATH.exists()
with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not exists:
        writer.writerow(headers)
    
    writer.writerow([
        team,
        run_id,
        score,
        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        author_type,
        model,
        notes
    ])
    
print(f"leaderboard updated {team}-{score}")
