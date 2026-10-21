"""Beginner NBA predictions: game winner, margin, total, player PTS/REB/AST.

Data modes:
  demo -> clearly simulated logs (to test the pipeline without internet)
  archive -> REAL game logs using a public hosted DuckDB archive (recommended in Colab)
  nba  -> real NBA regular-season game logs via nba_api (may timeout in Colab)
  csv  -> user-provided NBA logs in team_games.csv and player_games.csv

This is educational and not designed to provide guaranteed sports-betting edges.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error

SEASONS = ["2023-24", "2024-25", "2025-26"]
DATA_DIR = Path("nba_data")
TEAM_BASE = ["PTS", "OPP_PTS", "REB", "AST", "WIN"]
PLAYER_BASE = ["PTS", "REB", "AST", "MIN"]
TEAM_FEATURES = ([f"{v}_LAST5" for v in TEAM_BASE]
                 + [f"{v}_LAST10" for v in TEAM_BASE]
                 + ["REST_DAYS", "GAMES_BEFORE"])
PLAYER_FEATURES = ([f"P_{v}_LAST5" for v in PLAYER_BASE]
                   + [f"P_{v}_LAST10" for v in PLAYER_BASE]
                   + ["P_REST_DAYS", "P_GAMES_BEFORE", "IS_HOME", "OPP_ALLOWED_LAST5"])
GAME_FEATURES = ([f"H_{c}" for c in TEAM_FEATURES]
                 + [f"A_{c}" for c in TEAM_FEATURES]
                 + [f"DIFF_{c}" for c in TEAM_FEATURES])



# CHANGE ONLY THIS LINE TO START:
SOURCE = "archive"       # "archive" real NBA (recommended), "nba" direct, "demo" fictional, "csv" local
REFRESH_NBA_DATA = False   # Set True when you want to re-download NBA games

# nba_api is needed only for real NBA downloads.
if SOURCE == "nba":
    import importlib.util, subprocess, sys
    if importlib.util.find_spec("nba_api") is None:
        subprocess.check_call([sys.executable,"-m","pip","install","-q","nba_api"])
print("Selected data source:", SOURCE.upper())


def load_archive_data(seasons=SEASONS):
    """Download real regular-season game logs from an independent public NBA data archive.

    Source: https://github.com/llimllib/nba_data (v2 DuckDB catalog).
    Only historical games with two NBA teams and a known home court are included.
    """
    import importlib.util, subprocess, sys
    if importlib.util.find_spec("duckdb") is None:
        print("Installing duckdb for reading the NBA archive...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "duckdb"])
    import duckdb
    years = sorted({int(s[:4]) + 1 for s in seasons})
    allowed = "(" + ",".join(str(y) for y in years) + ")"
    print("Connecting to independent historical NBA archive; seasons ending", years)
    con = duckdb.connect()
    try:
        con.execute("ATTACH 'https://basketball-data.billmill.org/nba/nba.duckdb' AS nba;")
        teams = con.execute(f"""
            SELECT g.game_id AS GAME_ID,
                   g.game_date AS GAME_DATE,
                   own.nba_abbrev AS TEAM_ABBREVIATION,
                   CASE WHEN g.home THEN own.nba_abbrev || ' vs. ' || opp.nba_abbrev
                        ELSE own.nba_abbrev || ' @ ' || opp.nba_abbrev END AS MATCHUP,
                   g.pts AS PTS, g.plus_minus AS PLUS_MINUS, g.reb AS REB, g.ast AS AST,
                   CASE WHEN g.win THEN 'W' ELSE 'L' END AS WL
              FROM nba.team_game_logs g
              JOIN nba.games games ON games.season = g.season AND games.game_id = g.game_id
              JOIN nba.team_seasons own ON own.season = g.season AND own.team_id = g.team_id
              JOIN nba.team_seasons opp ON opp.season = g.season AND opp.team_id = g.opp_team_id
             WHERE g.season IN {allowed}
               AND games.game_type = 'regular_season'
               AND g.home IS NOT NULL
               AND own.is_nba = TRUE AND opp.is_nba = TRUE
        """).df()
        players = con.execute(f"""
            SELECT pg.game_id AS GAME_ID, pg.game_date AS GAME_DATE,
                   pg.player_id AS PLAYER_ID, names.name AS PLAYER_NAME,
                   own.nba_abbrev AS TEAM_ABBREVIATION,
                   CASE WHEN pg.home THEN own.nba_abbrev || ' vs. ' || opp.nba_abbrev
                        ELSE own.nba_abbrev || ' @ ' || opp.nba_abbrev END AS MATCHUP,
                   pg.min AS MIN, pg.pts AS PTS, pg.reb AS REB, pg.ast AS AST
              FROM nba.player_game_logs pg
              JOIN nba.games games ON games.season = pg.season AND games.game_id = pg.game_id
              JOIN nba.team_seasons own ON own.season = pg.season AND own.team_id = pg.team_id
              JOIN nba.team_seasons opp ON opp.season = pg.season AND opp.team_id = pg.opp_team_id
              JOIN nba.players names ON names.player_id = pg.player_id
             WHERE pg.season IN {allowed}
               AND games.game_type = 'regular_season'
               AND pg.home IS NOT NULL
               AND own.is_nba = TRUE AND opp.is_nba = TRUE
               AND pg.min > 0
        """).df()
    except Exception as err:
        raise RuntimeError(
            "The public NBA archive could not be loaded. Check Colab's internet connection "
            "and the archive URL, then retry. This does NOT mean demo mode is real data. "
            f"Original error: {type(err).__name__}: {err}"
        ) from err
    finally:
        con.close()
    # Keep actual paired games only (one home row + one away row).
    paired = teams.groupby("GAME_ID").filter(
        lambda group: len(group) == 2 and group.MATCHUP.str.contains("vs.", regex=False).sum() == 1
    )
    players = players[players.GAME_ID.isin(paired.GAME_ID)].copy()
    if len(paired) < 500 or len(players) < 5000:
        raise ValueError("Not enough usable REAL NBA logs were returned. No model has been trained.")
    print(f"REAL NBA DATA LOADED: {len(paired):,} team-game rows, {len(players):,} player-game rows")
    print("Latest historical game:", str(pd.to_datetime(paired.GAME_DATE).max())[:10])
    return paired, players


def _to_minutes(column):
    """Accept either decimal minutes or mm:ss strings from NBA endpoints."""
    s = column.astype(str)
    colon = s.str.contains(":", regex=False, na=False)
    numeric = pd.to_numeric(column, errors="coerce")
    if colon.any():
        parts = s[colon].str.split(":", expand=True)
        numeric.loc[colon] = (pd.to_numeric(parts[0], errors="coerce")
                              + pd.to_numeric(parts[1], errors="coerce") / 60)
    return numeric


def load_data(source="demo", data_dir=DATA_DIR, seasons=SEASONS, refresh=False):
    """Return NBA-style team and player per-game logs as DataFrames."""
    data_dir = Path(data_dir)
    if source == "demo":
        print("⚠ DEMONSTRATION ONLY: using generated, fictional games and players.")
        return make_demo_data()
    if source == "csv":
        print("Reading team_games.csv and player_games.csv from", data_dir.resolve())
        return (pd.read_csv(data_dir / "team_games.csv", dtype={"GAME_ID": str}),
                pd.read_csv(data_dir / "player_games.csv", dtype={"GAME_ID": str}))
    if source == "archive":
        return load_archive_data(seasons=seasons)
    if source != "nba":
        raise ValueError("source must be 'archive', 'demo', 'nba', or 'csv'")
    try:
        from nba_api.stats.endpoints import leaguegamelog, playergamelogs
    except ImportError as ex:
        raise ImportError("Install nba_api first: %pip install nba_api") from ex
    data_dir.mkdir(parents=True, exist_ok=True)
    teams, players = [], []
    for season in seasons:
        team_csv = data_dir / f"teams_{season}.csv"
        player_csv = data_dir / f"players_{season}.csv"
        if team_csv.exists() and player_csv.exists() and not refresh:
            print(f"Loading cached season {season}")
            t = pd.read_csv(team_csv, dtype={"GAME_ID": str})
            p = pd.read_csv(player_csv, dtype={"GAME_ID": str})
        else:
            print(f"Requesting real NBA {season} team games...")
            try:
                t = leaguegamelog.LeagueGameLog(
                    season=season, player_or_team_abbreviation="T",
                    season_type_all_star="Regular Season", timeout=45
                ).get_data_frames()[0]
                time.sleep(1.5)
                print(f"Requesting real NBA {season} player games...")
                p = playergamelogs.PlayerGameLogs(
                    season_nullable=season, season_type_nullable="Regular Season",
                    timeout=45
                ).get_data_frames()[0]
            except Exception as ex:
                raise RuntimeError(
                    f"Could not download NBA {season} stats: {type(ex).__name__}: {ex}. "
                    "The NBA sometimes blocks Colab. Switch source to 'demo' for "
                    "a working example or use source='csv' with real CSV exports."
                ) from ex
            if t.empty or p.empty:
                if season == seasons[-1] and len(teams) >= 2:
                    print(f"The latest season {season} has no results yet; using earlier seasons.")
                    continue
                raise RuntimeError(f"The NBA returned empty data for {season}; try a completed season.")
            t.to_csv(team_csv, index=False)
            p.to_csv(player_csv, index=False)
            time.sleep(1.5)
        teams.append(t)
        players.append(p)
    team_data = pd.concat(teams, ignore_index=True)
    player_data = pd.concat(players, ignore_index=True)
    print(f"Real data loaded: {len(team_data):,} team logs, {len(player_data):,} player logs")
    return team_data, player_data


def make_demo_data(seed=43, rounds=190):
    """Simulate a coherent small league. THESE ARE NOT REAL NBA GAMES."""
    rng = np.random.default_rng(seed)
    clubs = ["BOS", "NYK", "LAL", "GSW", "DEN", "MIA", "CHI", "DAL"]
    strength = {t: rng.normal(0, 7) for t in clubs}
    team_rows, player_rows = [], []
    dates = pd.date_range("2024-01-02", periods=rounds, freq="3D")
    for idx, day in enumerate(dates):
        shuffled = rng.permutation(clubs)
        for j in range(0, len(clubs), 2):
            home, away = shuffled[j], shuffled[j+1]
            gid = f"DEMO{idx:04d}{j:02d}"
            h_pts = max(70, round(112 + strength[home] - 0.50*strength[away] + 2.5 + rng.normal(0, 11)))
            a_pts = max(70, round(112 + strength[away] - 0.50*strength[home] + rng.normal(0, 11)))
            if h_pts == a_pts: h_pts += 2
            for team, opp, points, opp_points, at_home in [(home, away, h_pts, a_pts, True), (away, home, a_pts, h_pts, False)]:
                match = f"{team} vs. {opp}" if at_home else f"{team} @ {opp}"
                team_rows.append(dict(GAME_ID=gid, GAME_DATE=day, TEAM_ABBREVIATION=team,
                                      MATCHUP=match, PTS=points, PLUS_MINUS=points-opp_points,
                                      REB=max(25,int(rng.normal(44,7))), AST=max(10,int(rng.normal(26,5))),
                                      WL="W" if points>opp_points else "L"))
                for k in range(1,9):
                    player_id = clubs.index(team)*100+k
                    playing_time = max(5, rng.normal(33 - k*1.9, 3.5))
                    talent = 1.7 if k == 1 else 1.3 if k == 2 else 1.0 if k < 5 else .65
                    player_rows.append(dict(GAME_ID=gid, GAME_DATE=day, PLAYER_ID=player_id,
                                            PLAYER_NAME=f"Demo {team} Player {k}",
                                            TEAM_ABBREVIATION=team, MATCHUP=match,
                                            MIN=round(playing_time,1),
                                            PTS=max(0,round((points/8)*talent*(playing_time/25)+rng.normal(0,4))),
                                            REB=max(0,round(4*talent+rng.normal(0,2))),
                                            AST=max(0,round(3.3*talent+rng.normal(0,2)))))
    return pd.DataFrame(team_rows), pd.DataFrame(player_rows)


def clean_logs(team_raw, player_raw):
    team = team_raw.copy()
    player = player_raw.copy()
    team_required = {"GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP", "PTS", "PLUS_MINUS", "REB", "AST", "WL"}
    player_required = {"GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "MATCHUP", "PTS", "REB", "AST", "MIN"}
    for name, df, required in [("team", team, team_required), ("player", player, player_required)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} logs missing columns: {', '.join(sorted(missing))}")
        df["GAME_ID"] = df["GAME_ID"].astype(str)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.normalize()
        df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].astype(str).str.upper().str.strip()
        df.dropna(subset=["GAME_DATE"], inplace=True)
    for col in ("PTS", "PLUS_MINUS", "REB", "AST"):
        team[col] = pd.to_numeric(team[col], errors="coerce")
    for col in ("PTS", "REB", "AST"):
        player[col] = pd.to_numeric(player[col], errors="coerce")
    player["MIN"] = _to_minutes(player["MIN"])
    player["PLAYER_ID"] = player["PLAYER_ID"].astype(str)
    player["PLAYER_NAME"] = player["PLAYER_NAME"].astype(str)
    team = team.dropna(subset=["PTS", "PLUS_MINUS", "REB", "AST"])
    player = player.dropna(subset=["PTS", "REB", "AST", "MIN"])
    team = team.drop_duplicates(["GAME_ID", "TEAM_ABBREVIATION"])
    player = player.drop_duplicates(["GAME_ID", "PLAYER_ID"])
    team["OPP_PTS"] = team["PTS"] - team["PLUS_MINUS"]
    team["WIN"] = (team["WL"].astype(str).str.upper() == "W").astype(int)
    return team, player


def build_team_features(team):
    x = team.sort_values(["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"]).copy()
    g = x.groupby("TEAM_ABBREVIATION", sort=False)
    x["GAMES_BEFORE"] = g.cumcount()
    x["REST_DAYS"] = g["GAME_DATE"].diff().dt.days.clip(0,14)
    for col in TEAM_BASE:
        for n in (5,10):
            x[f"{col}_LAST{n}"] = g[col].transform(
                lambda s: s.shift(1).rolling(n, min_periods=3 if n == 5 else 5).mean())
    return x


def build_game_table(team_feat):
    # Use only true two-team games. Do NOT include points scored in model features.
    h = team_feat[team_feat["MATCHUP"].astype(str).str.contains("vs.", regex=False, na=False)].copy()
    a = team_feat[team_feat["MATCHUP"].astype(str).str.contains("@", regex=False, na=False)].copy()
    base = ["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "PTS"] + TEAM_FEATURES
    h = h[base].rename(columns={c: f"H_{c}" for c in base if c != "GAME_ID"})
    a = a[base].rename(columns={c: f"A_{c}" for c in base if c != "GAME_ID"})
    games = h.merge(a, on="GAME_ID", how="inner", validate="one_to_one")
    games = games[games["H_GAME_DATE"] == games["A_GAME_DATE"]].copy()
    games["GAME_DATE"] = games["H_GAME_DATE"]
    games["HOME_WIN"] = (games["H_PTS"] > games["A_PTS"]).astype(int)
    games["MARGIN"] = games["H_PTS"] - games["A_PTS"]
    games["TOTAL"] = games["H_PTS"] + games["A_PTS"]
    for c in TEAM_FEATURES:
        games[f"DIFF_{c}"] = games[f"H_{c}"] - games[f"A_{c}"]
    games = games[(games["H_GAMES_BEFORE"] >= 5) & (games["A_GAMES_BEFORE"] >= 5)].copy()
    return games.sort_values(["GAME_DATE", "GAME_ID"])


def build_player_features(player, team_feat):
    x = player.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).copy()
    g = x.groupby("PLAYER_ID", sort=False)
    x["P_GAMES_BEFORE"] = g.cumcount()
    x["P_REST_DAYS"] = g["GAME_DATE"].diff().dt.days.clip(0,14)
    for col in PLAYER_BASE:
        for n in (5,10):
            x[f"P_{col}_LAST{n}"] = g[col].transform(
                lambda s: s.shift(1).rolling(n, min_periods=3 if n == 5 else 5).mean())
    x["IS_HOME"] = x["MATCHUP"].str.contains("vs.", regex=False, na=False).astype(int)
    x["OPP_ABBR"] = x["MATCHUP"].str.extract(r"(?:vs\.\s*|@\s*)([A-Z]{2,3})", expand=False)
    # Correct, pregame opponent defensive rating (recent points conceded), no game-result leakage.
    defending = team_feat[["GAME_ID", "TEAM_ABBREVIATION", "OPP_PTS_LAST5"]].rename(
        columns={"TEAM_ABBREVIATION":"OPP_ABBR", "OPP_PTS_LAST5":"OPP_ALLOWED_LAST5"}
    )
    x = x.merge(defending, on=["GAME_ID", "OPP_ABBR"], how="left", validate="many_to_one")
    return x[x["P_GAMES_BEFORE"] >= 5].sort_values(["GAME_DATE", "PLAYER_ID"])


def _time_split(df, features, targets):
    df = df.dropna(subset=features + targets).sort_values("GAME_DATE").copy()
    days = np.sort(df["GAME_DATE"].unique())
    if len(days) < 20 or len(df) < 100:
        raise ValueError("Not enough historical games. Provide more completed seasons.")
    cut = pd.Timestamp(days[int(len(days)*0.8)])
    tr, te = df[df.GAME_DATE < cut], df[df.GAME_DATE >= cut]
    if len(tr)<70 or len(te)<20:
        raise ValueError("Not enough training/test rows for a useful backtest.")
    return tr,te,cut


def train_models(team_raw, player_raw):
    team, player = clean_logs(team_raw, player_raw)
    team_feat = build_team_features(team)
    games = build_game_table(team_feat)
    players = build_player_features(player, team_feat)
    gt, gv, gcut = _time_split(games, GAME_FEATURES, ["HOME_WIN", "MARGIN", "TOTAL"])
    pt, pv, pcut = _time_split(players, PLAYER_FEATURES, ["PTS", "REB", "AST"])
    # Do not use today's target results as input features.
    kwargs = dict(max_iter=110, max_leaf_nodes=15, min_samples_leaf=22,
                  l2_regularization=8.0, random_state=42)
    win = HistGradientBoostingClassifier(**kwargs).fit(gt[GAME_FEATURES],gt["HOME_WIN"])
    margin = HistGradientBoostingRegressor(**kwargs).fit(gt[GAME_FEATURES],gt["MARGIN"])
    total = HistGradientBoostingRegressor(**kwargs).fit(gt[GAME_FEATURES],gt["TOTAL"])
    pm = {stat:HistGradientBoostingRegressor(**kwargs).fit(pt[PLAYER_FEATURES],pt[stat])
          for stat in ("PTS","REB","AST")}
    # Walk-forward evaluation on the most recent 20% of dates.
    probs = win.predict_proba(gv[GAME_FEATURES])[:, list(win.classes_).index(1)]
    reports = {
        "game_test_games":len(gv), "game_test_since":str(gcut.date()),
        "winner_accuracy":accuracy_score(gv["HOME_WIN"],(probs>=0.5).astype(int)),
        "baseline_winner_accuracy":accuracy_score(gv["HOME_WIN"],
                                                    np.full(len(gv), int(gt["HOME_WIN"].mean() >= 0.5))),
        "winner_brier":brier_score_loss(gv["HOME_WIN"],probs),
        "margin_mae":mean_absolute_error(gv["MARGIN"], margin.predict(gv[GAME_FEATURES])),
        "baseline_margin_mae":mean_absolute_error(gv["MARGIN"], np.full(len(gv),gt["MARGIN"].mean())),
        "total_mae":mean_absolute_error(gv["TOTAL"], total.predict(gv[GAME_FEATURES])),
        "baseline_total_mae":mean_absolute_error(gv["TOTAL"], np.full(len(gv),gt["TOTAL"].mean())),
        "player_test_logs":len(pv), "player_test_since":str(pcut.date()),
    }
    for stat in ("PTS","REB","AST"):
        reports[f"player_{stat.lower()}_mae"] = mean_absolute_error(
            pv[stat],pm[stat].predict(pv[PLAYER_FEATURES]))
        reports[f"baseline_player_{stat.lower()}_mae"] = mean_absolute_error(
            pv[stat],pv[f"P_{stat}_LAST5"])
    # Backtest first; only after recording the test scores do we refit on ALL past games.
    # The refitted models are used for future predictions, not for scoring the holdout.
    full_games = games.dropna(subset=GAME_FEATURES+["HOME_WIN","MARGIN","TOTAL"])
    full_players = players.dropna(subset=PLAYER_FEATURES+["PTS","REB","AST"])
    win.fit(full_games[GAME_FEATURES],full_games["HOME_WIN"])
    margin.fit(full_games[GAME_FEATURES],full_games["MARGIN"])
    total.fit(full_games[GAME_FEATURES],full_games["TOTAL"])
    for stat in ("PTS","REB","AST"):
        pm[stat].fit(full_players[PLAYER_FEATURES],full_players[stat])
    return dict(models={"win":win,"margin":margin,"total":total,"player":pm},
                data={"team":team,"player":player},report=reports)


def show_report(state):
    r = state["report"]
    print("\n=== OUT-OF-SAMPLE TEST (newer games, not training games) ===")
    print(f"Game test: {r['game_test_games']:,} games since {r['game_test_since']}")
    print(f"Winner accuracy: {100*r['winner_accuracy']:.1f}% (simple baseline {100*r['baseline_winner_accuracy']:.1f}%)")
    print(f"Winner probability Brier: {r['winner_brier']:.3f} (lower better)")
    print(f"Home margin avg error: {r['margin_mae']:.1f} points (baseline {r['baseline_margin_mae']:.1f})")
    print(f"Combined total avg error: {r['total_mae']:.1f} points (baseline {r['baseline_total_mae']:.1f})")
    print(f"Player test: {r['player_test_logs']:,} logs since {r['player_test_since']}")
    for stat, name in (("pts","points"),("reb","rebounds"),("ast","assists")):
        print(f"Player {name} avg error: {r[f'player_{stat}_mae']:.1f} "
              f"(last-5-games baseline {r[f'baseline_player_{stat}_mae']:.1f})")
    print("Final prediction models were then retrained on all available completed games.")


def _prior_state(logs, identity_col, identity, day, vals, prefix=""):
    data = logs[(logs[identity_col].astype(str)==str(identity)) &
                (logs["GAME_DATE"]<day)].sort_values(["GAME_DATE","GAME_ID"])
    if len(data) < 5:
        raise ValueError(f"Not enough past games for {identity}: found {len(data)}, need 5.")
    state = {f"{prefix}{v}_LAST{n}":float(data[v].tail(n).mean())
             for v in vals for n in (5,10)}
    state[f"{prefix}REST_DAYS"] = float(min(14, max(0, (day-data.iloc[-1]["GAME_DATE"]).days)))
    state[f"{prefix}GAMES_BEFORE"] = len(data)
    return state


def _require_future_date(state, day):
    latest = state["data"]["team"]["GAME_DATE"].max()
    if day <= latest:
        raise ValueError(
            f"Prediction date must be AFTER latest training game ({latest.date()}). "
            "For historical predictions, retrain with only older games."
        )


def predict_game(state, home, away, game_date, home_spread=None, total_line=None):
    """home_spread: sportsbook convention, e.g. -4.5 if home favored by 4.5."""
    day = pd.Timestamp(game_date).normalize()
    _require_future_date(state, day)
    home, away = home.upper().strip(),away.upper().strip()
    if home == away: raise ValueError("Choose different teams.")
    team = state["data"]["team"]
    h = _prior_state(team,"TEAM_ABBREVIATION",home,day,TEAM_BASE)
    a = _prior_state(team,"TEAM_ABBREVIATION",away,day,TEAM_BASE)
    vals = {f"H_{c}":h[c] for c in TEAM_FEATURES}
    vals.update({f"A_{c}":a[c] for c in TEAM_FEATURES})
    vals.update({f"DIFF_{c}":h[c]-a[c] for c in TEAM_FEATURES})
    df = pd.DataFrame([vals],columns=GAME_FEATURES)
    m = state["models"]
    p_home=float(m["win"].predict_proba(df)[0,list(m["win"].classes_).index(1)])
    margin=float(m["margin"].predict(df)[0]); total=float(m["total"].predict(df)[0])
    result={"date":str(day.date()),"home":home,"away":away,
            "home_win_probability":round(p_home,3),"away_win_probability":round(1-p_home,3),
            "predicted_winner":home if p_home>=.5 else away,
            "home_margin":round(margin,1),"combined_total":round(total,1),
            "home_score":round((total+margin)/2,1),"away_score":round((total-margin)/2,1)}
    if (p_home >= .5) != (margin >= 0):
        result["warning"] = ("Winner and margin models disagree; "
                             "this is a particularly uncertain forecast.")
    if home_spread is not None:
        result["market_home_spread"] = home_spread
        result["spread_lean"] = (home+" to cover" if margin+home_spread>0 else
                                away+" to cover" if margin+home_spread<0 else "No edge")
    if total_line is not None:
        result["market_total"] = total_line
        result["total_lean"] = "Over" if total>total_line else "Under" if total<total_line else "No edge"
    return result


def predict_player(state, player_name, opponent, game_date, is_home=True,
                   points_line=None, rebounds_line=None, assists_line=None):
    """Conditional on player appearing in game; DOES NOT know injuries or minutes restrictions."""
    day = pd.Timestamp(game_date).normalize()
    _require_future_date(state, day)
    logs = state["data"]["player"]
    opponent=opponent.upper().strip()
    matches = logs[logs["PLAYER_NAME"].str.casefold() == player_name.casefold()]
    if matches.empty:
        choices=sorted(logs.loc[logs.PLAYER_NAME.str.contains(player_name,case=False,regex=False),"PLAYER_NAME"].unique())
        raise ValueError("Player not found. Closest names: "+", ".join(choices[:10]))
    ids = matches["PLAYER_ID"].unique()
    if len(ids)!=1:
        raise ValueError("Several players share that name; please use a unique player.")
    player_id=ids[0]
    p = _prior_state(logs,"PLAYER_ID",player_id,day,PLAYER_BASE,prefix="P_")
    p["IS_HOME"] = int(is_home)
    teams = state["data"]["team"]
    opp = _prior_state(teams,"TEAM_ABBREVIATION",opponent,day,TEAM_BASE)
    p["OPP_ALLOWED_LAST5"] = opp["OPP_PTS_LAST5"]
    df=pd.DataFrame([p],columns=PLAYER_FEATURES)
    models=state["models"]["player"]
    preds={stat:max(0,round(float(models[stat].predict(df)[0]),1)) for stat in ("PTS","REB","AST")}
    last = matches[matches.GAME_DATE<day].sort_values("GAME_DATE").iloc[-1]
    result={"date":str(day.date()),"player":last["PLAYER_NAME"],"team":last["TEAM_ABBREVIATION"],
            "opponent":opponent,"home":is_home,"projected_points":preds["PTS"],
            "projected_rebounds":preds["REB"],"projected_assists":preds["AST"]}
    for stat,line in [("PTS",points_line),("REB",rebounds_line),("AST",assists_line)]:
        if line is not None:
            result[f"{stat.lower()}_line"] = line
            result[f"{stat.lower()}_lean"] = "Over" if preds[stat]>line else "Under" if preds[stat]<line else "No edge"
    return result


def available_teams(state):
    return sorted(state["data"]["team"]["TEAM_ABBREVIATION"].unique())


def find_players(state, phrase):
    return sorted(state["data"]["player"].loc[
        lambda x: x.PLAYER_NAME.str.contains(phrase,case=False,regex=False),"PLAYER_NAME"].unique())[:30]

