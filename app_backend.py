"""Functions shared by the Streamlit UI and its self-contained tests.

Real historical data: public DuckDB NBA archive (same source as the Colab notebook).
Schedule: ESPN public scoreboard feed, used only for matchups/times.
"""
from __future__ import annotations

from datetime import date
from io import BytesIO

import pandas as pd

import nba_predictor_core as core
import nba_slate as slate

# nba_slate was originally a module loaded inside a Colab notebook, where the
# two prediction functions existed as global names. Attach the original backend
# prediction functions when loading it as a conventional Python module.
slate.predict_game = core.predict_game
slate.predict_player = core.predict_player


def demo_schedule(day: str) -> list[dict]:
    """Clearly synthetic games ONLY for testing the UI, never claimed as NBA schedule."""
    date.fromisoformat(day)
    return [
        {"game_id": "SIMULATED-1", "date": day, "tipoff_et": "07:00 PM ET", "away": "NYK", "home": "BOS"},
        {"game_id": "SIMULATED-2", "date": day, "tipoff_et": "08:30 PM ET", "away": "GSW", "home": "LAL"},
        {"game_id": "SIMULATED-3", "date": day, "tipoff_et": "10:00 PM ET", "away": "MIA", "home": "DEN"},
    ]


def get_schedule(day: str, source: str) -> list[dict]:
    date.fromisoformat(day)
    return demo_schedule(day) if source == "demo" else slate.fetch_schedule(day)


def build_state(day: str, source: str, team_csv: bytes | None = None,
                player_csv: bytes | None = None) -> dict:
    """Train only on games strictly before this day; never train on slate outcomes."""
    cutoff = pd.Timestamp(day).normalize()
    if source == "archive":
        seasons = slate.current_and_recent_seasons(day)
        teams, players = core.load_data(source="archive", seasons=seasons)
    elif source == "demo":
        teams, players = core.load_data(source="demo")
        # Shift made-up demonstration dates near the selected date so the demo
        # UI can show "recent" fictional players without mislabeling real data.
        last_demo_date = pd.to_datetime(teams["GAME_DATE"]).max().normalize()
        delta = (cutoff - pd.Timedelta(days=3)) - last_demo_date
        for logs in (teams, players):
            logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"]) + delta
    elif source == "csv":
        if not team_csv or not player_csv:
            raise ValueError("Upload both team and player historical CSV files.")
        teams = pd.read_csv(BytesIO(team_csv), dtype={"GAME_ID": str})
        players = pd.read_csv(BytesIO(player_csv), dtype={"GAME_ID": str})
    else:
        raise ValueError("Unknown source; use archive, csv, or demo.")
    if "GAME_DATE" not in teams or "GAME_DATE" not in players:
        raise ValueError("Historical CSVs must have a GAME_DATE column.")
    teams = teams[pd.to_datetime(teams["GAME_DATE"], errors="coerce") < cutoff].copy()
    players = players[pd.to_datetime(players["GAME_DATE"], errors="coerce") < cutoff].copy()
    if teams.empty or players.empty:
        raise ValueError("No historical games before this date. Pick a later date or upload earlier seasons.")
    state = core.train_models(teams, players)
    state["source"] = source
    state["cutoff_date"] = day
    return state


def predict_slate(state: dict, games: list[dict], day: str, players_per_team: int) -> tuple[pd.DataFrame,pd.DataFrame]:
    g, p = slate.produce_daily_predictions(state, games, day,
                                             top_players_per_team=players_per_team,
                                             include_players=players_per_team > 0)
    if len(g) and (g["prediction_status"] == "ERROR").all():
        raise ValueError("Every matchup prediction failed. Check team codes and training dates in the errors below:\n"
                         + "\n".join(str(x) for x in g["error"].head(3)))
    return g, p


def latest_training_date(state: dict) -> str:
    return str(state["data"]["team"]["GAME_DATE"].max().date())


def report_text(games: pd.DataFrame, players: pd.DataFrame, state: dict, day: str) -> str:
    r = state["report"]
    lines = [f"# NBA AI predictions — {day}", "",
             f"Source: {'SIMULATED (NOT REAL NBA)' if state['source'] == 'demo' else state['source']}",
             f"Last completed game in training: {latest_training_date(state)}",
             f"Historical holdout winner accuracy: {r['winner_accuracy']:.1%}", "",
             "## Upcoming matchups", ""]
    if games.empty:
        lines.append("No unstarted scheduled games on this date.")
    for row in games.to_dict("records"):
        game = f"{row.get('away')} @ {row.get('home')} ({row.get('tipoff_et')})"
        if row.get("prediction_status") != "OK":
            lines.append(f"- {game}: ERROR: {row.get('error')}")
        else:
            lines.append(f"- {game}: {row['predicted_winner']} projected winner; "
                         f"home-win probability {row['home_win_probability']:.1%}; "
                         f"forecast {row['away_score']:.1f}–{row['home_score']:.1f}; "
                         f"home margin {row['home_margin']:+.1f}; total {row['combined_total']:.1f}")
    lines.extend(["", f"Player projections generated: {len(players[players['prediction_status']=='OK']) if len(players) else 0}",
                  "", "Estimates only, not guaranteed outcomes. Player candidates are selected from "
                  "historical game logs, not confirmed active rosters. No live injury, lineup, "
                  "sportsbook line or actual betting-odds data is included."])
    return "\n".join(lines) + "\n"
