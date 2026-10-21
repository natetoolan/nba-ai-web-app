# NBA AI Forecast Lab — Web App

A beginner-friendly browser dashboard for the NBA machine-learning predictor you built in Google Colab. **This is the same original model**, now with a Streamlit user interface. The default source uses the public historical NBA archive (not `stats.nba.com`) and a live public scoreboard feed for upcoming games.

## What it does

- Pick a date on the **Eastern Time calendar** and click **Analyze this day's games**.
- See all **not-yet-started** NBA games returned by the schedule service, with predicted winners, win probabilities, scores, implied margins, and totals.
- View selected historical players' projections for points, rebounds and assists, **conditional on playing**.
- Enter any two NBA teams to calculate a custom matchup forecast; optionally supply your own sportsbook lines for a comparison.
- Download game predictions, player predictions, and a Markdown summary.
- Inspect historical holdout accuracy, average margin/total errors, and when the training data ends.
- Data source options: real public NBA game-log archive (default), your own CSVs, or a **clearly marked fictional demo** that is never substituted for real results.

## Option A: Put the web app online (recommended)

You'll need GitHub and Streamlit Community Cloud accounts. You do not need to leave your computer on once the site is deployed.

1. Download and **extract** `NBA_AI_Web_App.zip`.
2. At https://github.com/new create a repository such as `nba-ai-web-app` (for the easiest first deployment, a public repository; check that you are comfortable making the code public). Do not upload sensitive secrets or personal data.
3. In your new repository, select **Add file → Upload files** and upload these files from the extracted folder, preserving the names:
   - `app.py`
   - `app_backend.py`
   - `nba_predictor_core.py`
   - `nba_slate.py`
   - `daily_predictions.py`
   - `requirements.txt`
   - `README.md` (optional)
4. Click **Commit changes**.
5. Open https://share.streamlit.io/ and sign in with GitHub. Click **Create app**, select your repository and its default branch, and set **Main file path** to `app.py`. Click **Deploy**.
6. Open the resulting **`*.streamlit.app` link** in any browser. Pick today's date and click **Analyze this day's games**. Bookmark the link!
7. (Optional) Upload the extracted `.streamlit/config.toml` for the orange/dark theme. GitHub's browser upload may hide dotfolders; you can create the `.streamlit/config.toml` file directly in GitHub using **Add file → Create new file** and paste its contents.

**First real run:** The app must access the public archive and train models, so the initial run may take longer. The model is cached for up to 8 hours on a running Streamlit server. Free cloud apps may sleep or restart, so the cache can disappear. Streamlit Cloud hosting and external feeds have their own availability and resource limits.

### Adding the optional morning scheduler

This package also retains your previous GitHub workflow. If you want **unattended daily CSV reports**, create the file `.github/workflows/nba-daily.yml` in the same repository from the bundled file. Then follow the steps in the workflow and GitHub Actions tab. It runs `daily_predictions.py` separately and stores the report as a GitHub Actions artifact. **Those artifacts are not automatically loaded into the website**; the website runs fresh forecasts when you click Analyze. You can use either or both.

## Option B: Run as an app on your own computer

Install Python 3.11 or 3.12 from https://www.python.org/downloads/ (on Windows, make sure Python is added to PATH). Extract the ZIP and double-click `START_APP_WINDOWS.bat`, or open a terminal in the folder and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Mac/Linux use `python3` instead of `python`. This launches a local URL such as `http://localhost:8501` in your web browser. A local `localhost` site is **only accessible while your computer is on and running it**; use Option A for a website accessible from other devices.

## Files

- `app.py` — the complete Streamlit UI with four tabs.
- `app_backend.py` — the adapter that connects the UI to your existing model, enforces pregame-only training, and creates demo fixtures only in demo mode.
- `nba_predictor_core.py` — the same original classification/regression training and forecasting logic.
- `nba_slate.py` — schedule fetching, team-name mapping, historical player selection, and batch forecasts.
- `daily_predictions.py` and `.github/workflows/nba-daily.yml` — optional headless daily predictions.
- `requirements.txt` — packages automatically installed for deployment.
- `tests/test_backend.py` — offline checks for data leakage, fixtures, schedule filtering, and full demo predictions.

## Important data and model limitations

- The website is **not** the same as a bookmaker; predicted spread, winning margin, and total are **model projections**, not real-time sportsbook odds.
- The model's original holdout metrics are historical. They do not prove a reliable live betting advantage.
- Player choices are based on historical appearance and minutes and are **not a verified active roster**, and they do not account for injuries, changes in starting lineup or minutes restrictions.
- The public historical archive may lag behind completed NBA games, and the ESPN schedule feed is an undocumented endpoint that may change or fail. The app displays the newest data it actually trained on and refuses to silently replace real data with fictional games.
- The schedule only includes games still **not started** at the moment you request it. If a game has started or ended, it will not appear as an upcoming forecast; choose a future date for all upcoming games.
- For historical dates, it trains only on earlier games, but the current model setup is not a full time-accurate archive of rosters and injury reports. Do not treat a retrospective prediction as an independently validated backtest.
- For a commercial/public product, review third-party data usage terms and secure your service before publishing other users' personal data (none needed for this demo).

## Testing

The backend was tested with fictional input and the Python files were syntax-checked. Internet access to the external NBA sources and a deployed Streamlit session must be verified on your first run:

```bash
python -m pip install pytest
python -m pytest tests -q
```

For Streamlit hosting documentation: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
