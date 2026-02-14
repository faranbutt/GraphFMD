import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "leaderboard" / "leaderboard.csv"
MD_PATH = ROOT / "leaderboard" / "leaderboard.md"
HTML_PATH = ROOT / "docs" / "index.html"

def read_rows():
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if (r.get("team") or "").strip()]

def safe_float(val):
    try:
        if val is None or str(val).strip() == "" or str(val).strip().lower() == "none":
            return 0.0
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def main():
    rows = read_rows()
    rows.sort(key=lambda r: (-safe_float(r.get("macro_f1")), r.get("submitted_at", "")))

    ranked_rows = []
    current_rank = 1
    for i, row in enumerate(rows):
        if i > 0:
            prev_score = safe_float(rows[i-1].get("macro_f1"))
            curr_score = safe_float(row.get("macro_f1"))
            if curr_score != prev_score:
                current_rank = i + 1
        
        row["rank"] = current_rank
        ranked_rows.append(row)

    md_lines = ["# Leaderboard\n", "| Rank | Team | Score | Author | Model | Date (UTC) |\n", "|---:|---|---:|---|---|---|\n"]
    for r in ranked_rows:
        score_val = r.get('macro_f1') if r.get('macro_f1') else "0.0"
        md_lines.append(
            f"| {r['rank']} | {r['team']} | **{score_val}** | "
            f"{r.get('author_type', 'N/A')} | `{r.get('model', 'N/A')}` | {r.get('submitted_at', 'N/A')} |\n"
        )
    MD_PATH.write_text("".join(md_lines), encoding="utf-8")
    table_rows = []
    for r in ranked_rows:
        score_val = r.get('macro_f1') if r.get('macro_f1') else "0.0"
        table_rows.append(
            f"<tr><td>{r['rank']}</td><td>{r['team']}</td><td><strong>{score_val}</strong></td>"
            f"<td>{r.get('author_type', 'N/A')}</td><td>{r.get('model', 'N/A')}</td>"
            f"<td>{r.get('submitted_at', 'N/A')}</td></tr>"
        )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GraphFMD Leaderboard</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; background: #f4f4f9; color: #333; }}
            .container {{ max-width: 1000px; margin: auto; }}
            table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid #eee; }}
            th {{ background-color: #2c3e50; color: white; text-transform: uppercase; font-size: 14px; letter-spacing: 0.05em; }}
            tr:last-child td {{ border-bottom: none; }}
            tr:hover {{ background-color: #f9f9f9; }}
            h1 {{ color: #2c3e50; text-align: center; }}
            .timestamp {{ text-align: center; color: #666; font-size: 0.9em; margin-bottom: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>GraphFMD Competition Leaderboard</h1>
            <p class="timestamp">Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Team</th>
                        <th>Score (Macro F1)</th>
                        <th>Author</th>
                        <th>Model</th>
                        <th>Submitted At</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    HTML_PATH.parent.mkdir(exist_ok=True)
    HTML_PATH.write_text(html_content, encoding="utf-8")

if __name__ == "__main__":
    main()
