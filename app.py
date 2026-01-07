"""
IPL Fantasy Cricket Prediction Dashboard
=========================================
A Streamlit app for Dream11 team selection using ML predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = "dream11_predictions.csv"

# =============================================================================
# STEP 1: DATA LOADING (CACHED)
# =============================================================================
@st.cache_data
def load_data():
    """Load the predictions dataset."""
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data
def get_unique_teams(_df):
    """Extract unique teams for dropdowns."""
    teams = sorted(_df['Team'].dropna().unique().tolist())
    return teams


# =============================================================================
# STEP 2: DREAM TEAM SELECTION LOGIC
# =============================================================================
def get_match_players(df, team_a, team_b):
    """Get all players from both teams."""
    match_df = df[df['Team'].isin([team_a, team_b])].copy()

    # Remove duplicate players (keep highest predicted)
    match_df = match_df.sort_values('Predicted_Points', ascending=False)
    match_df = match_df.drop_duplicates(subset=['Player'], keep='first')

    return match_df.reset_index(drop=True)


def select_dream_team(predictions_df, top_n=11):
    """Select top N players and assign Captain/Vice-Captain."""

    if predictions_df.empty or len(predictions_df) < top_n:
        top_n = len(predictions_df)

    # Sort by predicted points and select top N
    dream_team = predictions_df.nlargest(top_n, 'Predicted_Points').reset_index(drop=True)

    # Assign roles
    dream_team['Role'] = ''
    if len(dream_team) >= 1:
        dream_team.loc[0, 'Role'] = 'CAPTAIN (2x)'
    if len(dream_team) >= 2:
        dream_team.loc[1, 'Role'] = 'VICE-CAPTAIN (1.5x)'

    # Calculate effective points
    dream_team['Effective_Pts'] = dream_team['Predicted_Points']
    if len(dream_team) >= 1:
        dream_team.loc[0, 'Effective_Pts'] = dream_team.loc[0, 'Predicted_Points'] * 2
    if len(dream_team) >= 2:
        dream_team.loc[1, 'Effective_Pts'] = dream_team.loc[1, 'Predicted_Points'] * 1.5

    dream_team['Effective_Pts'] = dream_team['Effective_Pts'].round(1)

    return dream_team


# =============================================================================
# STEP 3: STREAMLIT USER INTERFACE
# =============================================================================
def main():
    # Page config
    st.set_page_config(
        page_title="IPL Fantasy Predictor",
        page_icon="🏏",
        layout="wide"
    )

    # Header
    st.title("🏏 IPL Fantasy Cricket Predictor")
    st.markdown("*Build your winning Dream11 team using Machine Learning predictions*")
    st.divider()

    # Load data
    try:
        df = load_data()
        teams = get_unique_teams(df)
        st.success(f"Loaded {len(df)} player predictions!")
    except FileNotFoundError:
        st.error(f"Dataset file '{DATA_FILE}' not found!")
        return
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    # =========================
    # SIDEBAR - Match Selection
    # =========================
    with st.sidebar:
        st.header("⚙️ Match Setup")
        st.markdown("---")

        team_a = st.selectbox("🔵 Select Team A", options=teams, index=0)

        # Filter Team B to exclude Team A
        team_b_options = [t for t in teams if t != team_a]
        team_b = st.selectbox("🔴 Select Team B", options=team_b_options, index=0)

        st.markdown("---")
        predict_button = st.button("🎯 Predict Best XI", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.write(f"Total Predictions: **{len(df):,}**")
        st.write(f"Unique Players: **{df['Player'].nunique()}**")
        st.write(f"Teams: **{len(teams)}**")

    # =========================
    # MAIN SCREEN - Predictions
    # =========================
    if predict_button:
        # Get players from both teams
        match_players = get_match_players(df, team_a, team_b)

        if match_players.empty:
            st.warning("No players found for the selected teams!")
            return

        # Select Dream Team
        dream_team = select_dream_team(match_players, top_n=11)

        # Display Match Header
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown(f"### 🔵 {team_a}")
        with col2:
            st.markdown("### ⚔️ VS")
        with col3:
            st.markdown(f"### 🔴 {team_b}")

        st.divider()

        # Dream Team Display
        st.subheader("🏆 Your Dream11 Team")

        # Captain & Vice-Captain Cards
        if len(dream_team) >= 2:
            col1, col2 = st.columns(2)

            with col1:
                captain = dream_team.iloc[0]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #FFD700, #FFA500); padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: black; margin: 0;">👑 CAPTAIN</h2>
                    <h3 style="color: black; margin: 10px 0;">{captain['Player']}</h3>
                    <p style="color: black; margin: 0;"><strong>{captain['Team']}</strong></p>
                    <h4 style="color: black; margin: 10px 0;">{captain['Effective_Pts']} pts (2x)</h4>
                    <p style="color: black; font-size: 12px;">Career Avg: {captain['Career_Avg']} | Form: {captain['Recent_Form']}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                vc = dream_team.iloc[1]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #C0C0C0, #A9A9A9); padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: black; margin: 0;">🥈 VICE-CAPTAIN</h2>
                    <h3 style="color: black; margin: 10px 0;">{vc['Player']}</h3>
                    <p style="color: black; margin: 0;"><strong>{vc['Team']}</strong></p>
                    <h4 style="color: black; margin: 10px 0;">{vc['Effective_Pts']} pts (1.5x)</h4>
                    <p style="color: black; font-size: 12px;">Career Avg: {vc['Career_Avg']} | Form: {vc['Recent_Form']}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # Full Dream Team Table
        st.subheader("📋 Complete Squad")

        display_df = dream_team[['Player', 'Team', 'Role', 'Predicted_Points', 'Career_Avg', 'Recent_Form', 'Effective_Pts']].copy()
        display_df.columns = ['Player', 'Team', 'Role', 'Predicted Pts', 'Career Avg', 'Recent Form', 'Effective Pts']

        def highlight_roles(row):
            if row['Role'] == 'CAPTAIN (2x)':
                return ['background-color: #FFD700'] * len(row)
            elif row['Role'] == 'VICE-CAPTAIN (1.5x)':
                return ['background-color: #C0C0C0'] * len(row)
            return [''] * len(row)

        st.dataframe(
            display_df.style.apply(highlight_roles, axis=1),
            use_container_width=True,
            hide_index=True
        )

        # Total Predicted Points
        total_pts = dream_team['Effective_Pts'].sum()
        st.markdown(f"### 📈 Total Predicted Points: **{total_pts:.1f}**")

        st.divider()

        # Team Distribution
        st.subheader("📊 Team Distribution")
        team_counts = dream_team['Team'].value_counts()
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Players from {team_a}", team_counts.get(team_a, 0))
        with col2:
            st.metric(f"Players from {team_b}", team_counts.get(team_b, 0))

        # All Players Predictions
        with st.expander("📊 View All Match Players"):
            all_players = match_players.sort_values('Predicted_Points', ascending=False).reset_index(drop=True)
            st.dataframe(all_players, use_container_width=True, hide_index=True)

        # Risk Analysis - Players NOT in Dream11
        st.divider()
        st.subheader("⚠️ Risk Analysis - Players Left Out")

        dream_players = set(dream_team['Player'].tolist())
        left_out = match_players[~match_players['Player'].isin(dream_players)].copy()

        # High profile players left out (high career avg but low prediction)
        if not left_out.empty:
            left_out['Risk_Score'] = left_out['Career_Avg'] - left_out['Predicted_Points']
            risky_picks = left_out.nlargest(3, 'Risk_Score')

            if not risky_picks.empty:
                st.markdown("**High-profile players predicted to underperform:**")
                for _, player in risky_picks.iterrows():
                    if player['Career_Avg'] > player['Predicted_Points']:
                        st.markdown(f"""
                        - **{player['Player']}** ({player['Team']}): Career Avg **{player['Career_Avg']}** → Predicted **{player['Predicted_Points']:.1f}** pts
                          *Model suggests they may underperform their usual standards*
                        """)

    else:
        # Default state
        st.info("👈 Select teams from the sidebar, then click **Predict Best XI** to generate your Dream Team!")

        # Show sample data
        with st.expander("📄 Preview Dataset"):
            st.dataframe(df.head(15), use_container_width=True)


if __name__ == "__main__":
    main()
