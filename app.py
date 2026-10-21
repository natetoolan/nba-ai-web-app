"""NBA AI Forecast Lab. Run with: streamlit run app.py"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import app_backend as backend
import nba_predictor_core as core

st.set_page_config(page_title="NBA AI Forecast Lab", page_icon="🏀", layout="wide",
                   initial_sidebar_state="expanded")

# Native Streamlit controls remain usable on desktop, tablet, and mobile.
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1260px;}
div[data-testid="stMetric"] {background:#172235; border:1px solid #26364d;
  border-radius:13px; padding:15px 18px;}
small, .stCaption {opacity:.86;}
</style>
""", unsafe_allow_html=True)

ET = ZoneInfo("America/New_York")
today = datetime.now(ET).date()

@st.cache_data(ttl=12 * 60, max_entries=12, show_spinner=False)
def cached_schedule(day: str, source: str) -> list[dict]:
    return backend.get_schedule(day, source)

@st.cache_resource(ttl=8 * 3600, max_entries=4, show_spinner=False)
def cached_model(day: str, source: str) -> dict:
    # Refreshes after 8 hours; can take longer on the first visitor.
    return backend.build_state(day, source)


st.markdown("### 🏀 NBA AI · Forecast Lab")
st.caption("Your existing NBA machine-learning model, now in a single dashboard · US Eastern game dates")

with st.sidebar:
    st.header("Prediction settings")
    selected_day = st.date_input("Game date (Eastern)", value=today,
                                  help="For a future game date, the model only trains on earlier games.")
    mode = st.selectbox("Historical data", ["Real NBA archive", "Upload my CSVs", "Demo — simulated only"],
                        help="Real archive uses your working historical data source, not the NBA stats endpoint.")
    source = {"Real NBA archive": "archive", "Upload my CSVs": "csv", "Demo — simulated only": "demo"}[mode]
    players_per_team = st.slider("Player projections per team", 0, 10, 5,
                                help="Historical player candidates, not a verified active roster.")
    teams_file = players_file = None
    if source == "csv":
        st.caption("Use the same team_games.csv and player_games.csv layout as your original notebook.")
        teams_file = st.file_uploader("Historical team game logs", type=["csv"], key="teams_upload")
        players_file = st.file_uploader("Historical player game logs", type=["csv"], key="players_upload")
    if source == "demo":
        st.warning("Demo uses made-up results and a fictional 3-game slate. Never treat these as real NBA predictions.")
    run_clicked = st.button("🏀 Analyze this day's games", type="primary", use_container_width=True)
    st.caption("Nothing is posted to a sportsbook or a betting account.")
    st.divider()
    st.caption("**Where results come from**")
    st.caption("Game schedule: ESPN public scoreboard feed (upcoming only). Historical statistics: independently hosted NBA game-log archive. Source services can be unavailable or delayed.")

selected_date = selected_day.isoformat()
config = (selected_date, source, players_per_team)

if run_clicked:
    if source == "csv" and not (teams_file and players_file):
        st.error("Please upload BOTH historical game-log CSVs, or choose Real NBA archive.")
    else:
        try:
            with st.spinner(f"Finding scheduled games for {selected_date}..."):
                games = cached_schedule(selected_date, source)
            if not games:
                # This is a valid no-games condition, not a model failure.
                st.session_state["forecast"] = dict(config=config, schedule=[], games=pd.DataFrame(),
                                                     players=pd.DataFrame(), state=None)
            else:
                with st.spinner("Loading historical statistics and training your prediction models..."):
                    if source == "csv":
                        model_state = backend.build_state(selected_date, source,
                                                          teams_file.getvalue(), players_file.getvalue())
                    else:
                        model_state = cached_model(selected_date, source)
                with st.spinner("Predicting the full game schedule and player statistics..."):
                    game_df, player_df = backend.predict_slate(model_state, games, selected_date,
                                                               players_per_team)
                st.session_state["forecast"] = dict(config=config, schedule=games,
                                                     games=game_df, players=player_df,
                                                     state=model_state)
        except Exception as exc:
            st.error(f"Unable to complete predictions: {type(exc).__name__}: {exc}")
            st.info("If the real archive or schedule service is unavailable, try again later or upload your own historical CSVs. The app never substitutes made-up data for real predictions.")

forecast = st.session_state.get("forecast")
if not forecast or forecast["config"] != config:
    st.info("Choose your date and click **Analyze this day's games** to generate real forecasts. Your model will be trained using games before that date.")
    a, b, c = st.columns(3)
    a.metric("Mode", "Daily slate")
    b.metric("Predictions", "Game + player")
    c.metric("Data", "Historical")
    st.markdown("#### What you'll see")
    st.write("Every upcoming NBA matchup on the selected day, predicted win probabilities, final scores, estimated spreads and totals, selected player points/rebounds/assists, and CSV downloads.")
    st.stop()

schedule = forecast["schedule"]
game_df = forecast["games"]
player_df = forecast["players"]
state = forecast["state"]
if source == "demo":
    st.error("SIMULATED DEMO — these are NOT real NBA games, predictions or odds.")
if not schedule:
    st.success(f"No unstarted NBA matchups returned for {selected_date}.")
    st.caption("Games already underway or finished are deliberately excluded. Check a future date to see the complete upcoming slate.")
    st.stop()

ok_games = game_df[game_df["prediction_status"] == "OK"].copy()
ok_players = player_df[player_df["prediction_status"] == "OK"].copy() if not player_df.empty else pd.DataFrame()
last_game = backend.latest_training_date(state)
report = state["report"]

st.subheader(f"Daily forecasts · {selected_day.strftime('%b %d, %Y')}")
metrics = st.columns(4)
metrics[0].metric("Scheduled / unstarted", len(schedule))
metrics[1].metric("Forecasts generated", len(ok_games))
metrics[2].metric("Holdout winner accuracy", f"{report['winner_accuracy']:.1%}")
metrics[3].metric("Last game in training", last_game)
st.caption("Holdout accuracy is historical, not your likelihood of winning a bet. Predictions may be stale if the archive has not added recent games.")

TAB_GAMES, TAB_PLAYERS, TAB_CUSTOM, TAB_MODEL = st.tabs(
    ["🏆 Today's games", "📊 Player projections", "🎯 Custom matchup", "🔬 Model quality"])

with TAB_GAMES:
    st.markdown("#### Predictions for every upcoming game")
    st.caption("Projected spreads are the model's expected margins. Sportsbook spreads and totals are NOT fetched.")
    for row in game_df.to_dict("records"):
        with st.container(border=True):
            header, time_col = st.columns([4, 1])
            header.markdown(f"**{row['away']} @ {row['home']}**")
            time_col.caption(row.get("tipoff_et", ""))
            if row.get("prediction_status") != "OK":
                st.warning(row.get("error", "This prediction failed."))
                continue
            p_home = float(row["home_win_probability"])
            winner = row["predicted_winner"]
            confidence = p_home if winner == row["home"] else (1 - p_home)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Projected winner", winner)
            c2.metric("Win probability", f"{confidence:.1%}")
            c3.metric("Predicted score", f"{row['away_score']:.0f} – {row['home_score']:.0f}")
            c4.metric("Total points", f"{row['combined_total']:.1f}")
            st.progress(p_home, text=f"{row['home']} win chance: {p_home:.1%}  ·  {row['away']} win chance: {1-p_home:.1%}")
            st.caption(f"Model's home winning margin: {float(row['home_margin']):+.1f} points (positive = home favored)")
            if row.get("warning"):
                st.warning(row["warning"])
    st.divider()
    st.markdown("#### Export predictions")
    a,b,c = st.columns(3)
    a.download_button("Download game forecasts (CSV)", game_df.to_csv(index=False).encode("utf-8"),
                      f"nba_games_{selected_date}.csv", mime="text/csv", use_container_width=True)
    b.download_button("Download player forecasts (CSV)", player_df.to_csv(index=False).encode("utf-8"),
                      f"nba_players_{selected_date}.csv", mime="text/csv", use_container_width=True)
    c.download_button("Download summary (Markdown)", backend.report_text(game_df,player_df,state,selected_date),
                      f"nba_report_{selected_date}.md", mime="text/markdown", use_container_width=True)
    with st.expander("All game forecasts as a table"):
        st.dataframe(game_df, use_container_width=True, hide_index=True)

with TAB_PLAYERS:
    st.markdown("#### Points · Rebounds · Assists")
    st.warning("Players here are selected using recent historical games; this DOES NOT check today's active roster, injuries, or confirmed minutes. Projections assume the player plays.")
    if ok_players.empty:
        st.info("No eligible historical player projections for the selected slate. Try increasing players per team or refreshing historical data.")
    else:
        matchup_names = list(dict.fromkeys(
            f"{g['away']} @ {g['home']}" for g in game_df.to_dict("records") if g.get("prediction_status") == "OK"
        ))
        match = st.selectbox("Filter by matchup", ["All games"] + matchup_names)
        view = ok_players
        if match != "All games":
            away, home = match.split(" @ ")
            view = view[view["team"].isin([away, home])]
        ordered = ["player", "team", "opponent", "projected_points", "projected_rebounds", "projected_assists", "recent_minutes"]
        available = [c for c in ordered if c in view.columns]
        st.dataframe(view[available].sort_values("projected_points",ascending=False),
                     hide_index=True, use_container_width=True)
        st.caption("Projected statistics are estimates, not confirmed player availability or prop odds.")

with TAB_CUSTOM:
    st.markdown("#### Predict any matchup with your trained model")
    names = core.available_teams(state)
    left, right = st.columns(2)
    with left:
        home = st.selectbox("Home team", names, index=names.index("BOS") if "BOS" in names else 0)
    with right:
        away_choices = [t for t in names if t != home]
        away = st.selectbox("Away team", away_choices,
                            index=away_choices.index("NYK") if "NYK" in away_choices else 0)
    with st.expander("Optional: compare to a sportsbook line you enter yourself"):
        line_enabled = st.checkbox("Enter market lines", value=False)
        c1, c2 = st.columns(2)
        with c1:
            market_spread = st.number_input("Home spread (e.g. -3.5)", value=-3.5, step=.5,
                                            disabled=not line_enabled)
        with c2:
            market_total = st.number_input("Over/under total", value=225.5, step=.5,
                                           disabled=not line_enabled)
    if st.button("Predict custom matchup", use_container_width=True):
        try:
            pred = core.predict_game(state, home, away, selected_date,
                                     home_spread=market_spread if line_enabled else None,
                                     total_line=market_total if line_enabled else None)
            st.success(f"Predicted winner: {pred['predicted_winner']}")
            c1,c2,c3=st.columns(3)
            c1.metric("Home win chance",f"{pred['home_win_probability']:.1%}")
            c2.metric("Projected score", f"{away} {pred['away_score']:.0f} · {home} {pred['home_score']:.0f}")
            c3.metric("Combined points",pred["combined_total"])
            if line_enabled:
                st.write(f"Compared with entered lines: spread = {pred.get('spread_lean')}; total = {pred.get('total_lean')} (not wagering advice).")
            if pred.get("warning"):
                st.warning(pred["warning"])
        except Exception as exc:
            st.error(f"Custom matchup failed: {exc}")
    st.divider()
    st.markdown("#### Look up any historical player")
    player_names = sorted(state["data"]["player"]["PLAYER_NAME"].dropna().unique())
    player_name = st.selectbox("Player in historical dataset", player_names)
    o1,o2 = st.columns(2)
    with o1:
        opponent = st.selectbox("Opponent", names, key="player_opp")
    with o2:
        is_home = st.checkbox("Player's team is at home", value=True)
    if st.button("Predict player stats"):
        try:
            outcome = core.predict_player(state, player_name, opponent, selected_date, is_home)
            st.caption(f"Last historically recorded team: {outcome['team']} · assumes participation")
            cc = st.columns(3)
            cc[0].metric("Points", outcome["projected_points"])
            cc[1].metric("Rebounds", outcome["projected_rebounds"])
            cc[2].metric("Assists", outcome["projected_assists"])
        except Exception as exc:
            st.error(str(exc))

with TAB_MODEL:
    st.markdown("#### How well did the model perform on historical held-out games?")
    r=state["report"]
    a,b,c=st.columns(3)
    a.metric("Winner accuracy",f"{r['winner_accuracy']:.1%}")
    b.metric("Average margin error",f"{r['margin_mae']:.1f} pts")
    c.metric("Average total error",f"{r['total_mae']:.1f} pts")
    st.write(f"The evaluation used {r['game_test_games']:,} held-out matchups beginning {r['game_test_since']}. A constant majority-class baseline achieved {r['baseline_winner_accuracy']:.1%} winner accuracy.")
    st.write(f"Historical player stat average errors: points **{r['player_pts_mae']:.1f}**, rebounds **{r['player_reb_mae']:.1f}**, assists **{r['player_ast_mae']:.1f}**.")
    st.caption("The original model was retrained on all eligible earlier data after its holdout evaluation. Historical holdout scores are not prospective live performance; audit this system over time before relying on it.")
    st.divider()
    st.write(f"**Historical data mode:** {source}")
    st.write(f"**Latest completed game in dataset:** {last_game}")
    st.write(f"**Prediction date:** {selected_date} (US Eastern)")
    st.write(f"**Historical team-game rows:** {len(state['data']['team']):,}")
    st.write(f"**Historical player-game rows:** {len(state['data']['player']):,}")
    st.caption("The app does not automatically verify injuries, recent trades, starting lineups, sportsbooks or live odds. Availability of free third-party data sources is not guaranteed.")
