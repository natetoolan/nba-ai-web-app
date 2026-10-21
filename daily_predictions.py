"""Run all scheduled NBA forecasts for an Eastern-calendar date, once per execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import nba_predictor_core as core
import nba_slate as slate

# The batch runner calls the SAME prediction functions as the Colab notebook.
slate.predict_game = core.predict_game
slate.predict_player = core.predict_player


def run(day: str | None = None, *, source="archive", out_dir="outputs", top_players=5):
    day = day or slate.today_eastern()
    day_ts = pd.Timestamp(day).normalize()
    schedule = slate.fetch_schedule(day)
    print(f"NBA schedule for {day}: {len(schedule)} not-yet-started games")
    if source == "demo":
        print("CAUTION: DEMO uses SIMULATED historical scores and players. Not real predictions!")
    # If there aren't any games, avoid downloading the historical archive.
    if not schedule:
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        game_csv = output / f"nba_games_{day}.csv"
        player_csv = output / f"nba_players_{day}.csv"
        report = output / f"nba_report_{day}.md"
        pd.DataFrame(columns=["date", "tipoff_et", "away", "home", "predicted_winner"]).to_csv(game_csv,index=False)
        pd.DataFrame(columns=["date", "player", "team", "opponent"]).to_csv(player_csv,index=False)
        report.write_text(f"# NBA AI daily report — {day}\n\nNo upcoming scheduled NBA games returned for this Eastern-calendar date.\n", encoding="utf-8")
        print("Nothing to predict today. Empty reports saved.")
        return {"games": str(game_csv), "players": str(player_csv), "report": str(report)}
    seasons = slate.current_and_recent_seasons(day)
    print("Loading historical NBA seasons:", ", ".join(seasons))
    core.SEASONS = seasons
    t, p = core.load_data(source=source, seasons=seasons, data_dir=core.DATA_DIR)
    # Prevent leakage: the archive could include game results from the SAME day.
    # Data from that day must NEVER be available for same-day pregame predictions.
    t = t[pd.to_datetime(t["GAME_DATE"]) < day_ts].copy()
    p = p[pd.to_datetime(p["GAME_DATE"]) < day_ts].copy()
    if t.empty or p.empty:
        raise RuntimeError("No completed games before slate date; refusing to fabricate predictions.")
    state = core.train_models(t, p)
    core.show_report(state)
    print("Latest game included:", state["data"]["team"].GAME_DATE.max().date())
    game_predictions, player_predictions = slate.produce_daily_predictions(
        state, schedule, day, top_players_per_team=top_players)
    slate.show_slate(game_predictions, player_predictions, day)
    paths = slate.save_slate_reports(game_predictions, player_predictions, day, state, out_dir)
    print("Files saved:\n"+json.dumps(paths, indent=2))
    if not game_predictions.empty and (game_predictions["prediction_status"] != "OK").all():
        raise RuntimeError("No scheduled matchup could be predicted; see errors in the games CSV file.")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily automatic NBA model predictions")
    parser.add_argument("--date", default="", help="YYYY-MM-DD (blank means today in US Eastern time)")
    parser.add_argument("--source", choices=["archive", "csv", "demo"], default="archive")
    parser.add_argument("--output", default="outputs", help="Folder for CSV and Markdown outputs")
    parser.add_argument("--top-players", type=int, default=5, help="Historical players per team, not verified roster")
    args = parser.parse_args()
    run(args.date or None, source=args.source, out_dir=args.output, top_players=args.top_players)
