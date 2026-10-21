"""Schedule loading, batch predictions, and reporting for the NBA AI notebook.

ESPN's site.api.espn.com scoreboard feed is an undocumented public feed;
requests may fail or its schema may change. No model estimates use ESPN's
score predictions; only the date, home/away teams, and tip time are consumed.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json

import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
TEAM_ALIASES = {"NY": "NYK", "GS": "GSW", "SA": "SAS", "NO": "NOP", "WSH": "WAS",
                "UT": "UTA", "UTAH": "UTA", "PHO": "PHX", "BRK": "BKN"}
DEFAULT_TOP_PLAYERS_PER_TEAM = 5


def today_eastern() -> str:
    """NBA slate dates use the calendar date in US Eastern time."""
    return datetime.now(ET).strftime("%Y-%m-%d")


def normalize_team(team: str) -> str:
    short = str(team).strip().upper()
    return TEAM_ALIASES.get(short, short)


def current_and_recent_seasons(on_date: str | None = None, count: int = 4) -> list[str]:
    """E.g., Oct 2026 -> ['2023-24', '2024-25', '2025-26', '2026-27']."""
    d = date.fromisoformat(on_date or today_eastern())
    end_year = d.year + 1 if d.month >= 7 else d.year
    return [f"{y - 1}-{y % 100:02d}" for y in range(end_year - count + 1, end_year + 1)]


def parse_schedule(payload: dict, game_date: str) -> list[dict]:
    """Return only PRE-GAME scheduled NBA matchups on the requested ET date.

    Cannot forecast in-progress or completed games as if they were upcoming.
    Empty events are a real no-games result; malformed responses raise.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError("Schedule response is missing an 'events' list.")
    target = date.fromisoformat(game_date)
    found: list[dict] = []
    for event in payload["events"]:
        competition = (event.get("competitions") or [{}])[0]
        status = competition.get("status") or event.get("status") or {}
        stype = status.get("type") or {}
        state = str(stype.get("state") or "").lower()
        status_name = str(stype.get("name") or "").upper()
        if state != "pre" or any(skip in status_name for skip in ("POSTPONED", "CANCELED", "CANCELLED")):
            continue
        time_iso = event.get("date") or competition.get("date")
        if not time_iso:
            raise ValueError(f"A scheduled event {event.get('id')} is missing its tipoff time.")
        tipoff = datetime.fromisoformat(time_iso.replace("Z", "+00:00")).astimezone(ET)
        if tipoff.date() != target:
            continue
        sides = {}
        for item in competition.get("competitors", []):
            side = item.get("homeAway")
            if side in {"home", "away"}:
                code = item.get("team", {}).get("abbreviation", "")
                if not code:
                    raise ValueError(f"Missing team abbreviation for event {event.get('id')}")
                sides[side] = normalize_team(code)
        if set(sides) != {"home", "away"}:
            raise ValueError(f"Missing home or away team for event {event.get('id')}")
        found.append({"game_id": str(event.get("id", "")),
                      "date": game_date, "tipoff_et": tipoff.strftime("%I:%M %p ET"),
                      "home": sides["home"], "away": sides["away"]})
    found.sort(key=lambda x: (datetime.strptime(x["tipoff_et"], "%I:%M %p ET"), x["game_id"]))
    return found


def fetch_schedule(game_date: str, *, session=None, timeout=25) -> list[dict]:
    date.fromisoformat(game_date)
    requester = session if session is not None else requests
    try:
        response = requester.get(SCOREBOARD_URL, params={"dates": game_date.replace("-", ""), "limit": 100}, timeout=timeout)
        response.raise_for_status()
        games = parse_schedule(response.json(), game_date)
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise RuntimeError(f"Unable to load NBA schedule for {game_date} from ESPN: {exc}") from exc
    return games


def select_historical_players(state: dict, team: str, game_date: str,
                              count: int = DEFAULT_TOP_PLAYERS_PER_TEAM,
                              max_age_days: int = 260) -> list[dict]:
    """Candidates only: latest historical team, not verified active roster/injuries."""
    log = state["data"]["player"].copy()
    day = pd.Timestamp(game_date)
    log = log[log["GAME_DATE"] < day].sort_values(["GAME_DATE", "GAME_ID"])
    if log.empty:
        return []
    # By ID, not player name (names are sometimes duplicated).
    latest = log.groupby("PLAYER_ID", sort=False).tail(1)
    cutoff = day - pd.Timedelta(days=max_age_days)
    latest = latest[(latest["TEAM_ABBREVIATION"].astype(str).str.upper() == team)
                    & (latest["GAME_DATE"] >= cutoff)]
    scored = []
    for row in latest.itertuples(index=False):
        history = log.loc[log["PLAYER_ID"] == row.PLAYER_ID]
        if len(history) < 5:
            continue
        mins = float(pd.to_numeric(history["MIN"].tail(5), errors="coerce").mean())
        if pd.notna(mins) and mins >= 15:
            scored.append({"name": row.PLAYER_NAME, "minutes_recent": round(mins, 1)})
    return sorted(scored, key=lambda p: p["minutes_recent"], reverse=True)[:count]


def produce_daily_predictions(state: dict, slate: list[dict], game_date: str,
                              *, top_players_per_team: int = DEFAULT_TOP_PLAYERS_PER_TEAM,
                              include_players: bool = True):
    """Use the functions in the existing trained NBA notebook to predict a slate.

    Caller must make predict_game / predict_player available in this module's
    namespace (as exported by nba_predictor_core in the standalone version).
    """
    game_rows, player_rows = [], []
    for scheduled in slate:
        home, away = scheduled["home"], scheduled["away"]
        base = {"date": game_date, "tipoff_et": scheduled["tipoff_et"],
                "away": away, "home": home, "game_id": scheduled.get("game_id", "")}
        try:
            pred = predict_game(state, home, away, game_date)
            game_rows.append({**base, **pred, "prediction_status": "OK", "error": ""})
        except Exception as exc:
            game_rows.append({**base, "prediction_status": "ERROR", "error": str(exc)})
            continue
        if not include_players:
            continue
        for team, opponent, at_home in ((home, away, True), (away, home, False)):
            candidates = select_historical_players(state, team, game_date,
                                                    count=top_players_per_team)
            for candidate in candidates:
                try:
                    pp = predict_player(state, candidate["name"], opponent,
                                        game_date, is_home=at_home)
                    player_rows.append({"date": game_date, "tipoff_et": scheduled["tipoff_et"],
                                        "game_id": scheduled.get("game_id", ""),
                                        "recent_minutes": candidate["minutes_recent"],
                                        **pp, "prediction_status": "OK", "error": ""})
                except Exception as exc:
                    player_rows.append({"date": game_date, "game_id": scheduled.get("game_id", ""),
                                        "player": candidate["name"], "team": team,
                                        "opponent": opponent, "prediction_status": "ERROR", "error": str(exc)})
    game_df = pd.DataFrame(game_rows)
    player_df = pd.DataFrame(player_rows)
    if player_df.empty:
        player_df = pd.DataFrame(columns=["date", "game_id", "player", "team", "opponent",
                                          "projected_points", "projected_rebounds", "projected_assists",
                                          "prediction_status", "error"])
    return game_df, player_df


def save_slate_reports(games: pd.DataFrame, players: pd.DataFrame, game_date: str,
                       state: dict, out_dir="outputs") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    game_file = out / f"nba_games_{game_date}.csv"
    player_file = out / f"nba_players_{game_date}.csv"
    md_file = out / f"nba_report_{game_date}.md"
    games.to_csv(game_file, index=False)
    players.to_csv(player_file, index=False)
    latest = str(state["data"]["team"]["GAME_DATE"].max().date())
    accuracy = float(state["report"]["winner_accuracy"])*100
    heading = [f"# NBA AI daily report — {game_date}", "",
               f"Generated: {datetime.now(ET).strftime('%Y-%m-%d %I:%M %p ET')}",
               f"Latest game in training data: {latest}",
               f"Previous holdout winner accuracy: {accuracy:.1f}% (historical, not a promise)",
               "", "## Matchups", ""]
    if games.empty:
        heading.append("No upcoming scheduled games were returned for this date.")
    else:
        for game in games.to_dict(orient="records"):
            matchup = f"{game['away']} @ {game['home']} ({game['tipoff_et']})"
            if game.get("prediction_status") != "OK":
                heading.append(f"- **{matchup}:** ERROR — {game.get('error', '')}")
            else:
                home_win = float(game["home_win_probability"]) * 100
                heading.append(
                    f"- **{matchup}:** {game['predicted_winner']} predicted winner; "
                    f"home win {home_win:.1f}%; projected score "
                    f"{game['away']} {game['away_score']} – {game['home']} {game['home_score']}; "
                    f"home margin {game['home_margin']:+.1f}; game total {game['combined_total']:.1f}")
    heading.extend(["", "## Notes", "",
      "- Projections are model estimates, not guaranteed outcomes or sportsbook odds.",
      "- No live injury checks, confirmed starters, or lineups are included.",
      "- Player selections are historical team candidates, not verified current rosters.",
      "- Player projections assume the player participates in the game.",
      "- Score and margin models can disagree with the winner probability model."])
    md_file.write_text("\n".join(heading) + "\n", encoding="utf-8")
    return {"games": str(game_file), "players": str(player_file), "report": str(md_file)}


def show_slate(games: pd.DataFrame, players: pd.DataFrame, game_date: str):
    print(f"\n=== NBA AI DAILY SLATE: {game_date} ===")
    if games.empty:
        print("No upcoming scheduled games found for this date.")
    else:
        view = ["tipoff_et", "away", "home", "predicted_winner", "home_win_probability",
                "away_score", "home_score", "home_margin", "combined_total", "prediction_status"]
        print(games.reindex(columns=view).to_string(index=False))
    if not players.empty:
        print(f"\nPlayer projections generated: {int((players.prediction_status == 'OK').sum())}")
        print("(These players are historical candidates — availability/rosters not verified.)")
    else:
        print("No recent historical player candidates met the selection rules; no player projections.")

