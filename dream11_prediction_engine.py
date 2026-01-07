"""
Dream11 IPL Fantasy Points Prediction Engine
=============================================
A production-ready ML pipeline for predicting player fantasy points in IPL matches.

Features:
- Historical performance analysis (2008-2024)
- Rolling averages with no data leakage
- Venue-specific and opposition-specific stats
- XGBoost/RandomForest model training
- Dream11 team recommendation system

Author: IPL Oracle
Dataset: IPL_ball_by_ball_updated.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

# Try to import XGBoost (optional, falls back to RandomForest)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Using RandomForest instead.")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Fantasy Points Configuration (Dream11 style)
POINTS_CONFIG = {
    'run': 1,                    # 1 point per run
    'boundary_bonus': 1,         # +1 for hitting a four
    'six_bonus': 2,              # +2 for hitting a six
    'wicket': 25,                # 25 points per wicket
    'lbw_bowled_bonus': 8,       # +8 bonus for LBW/Bowled
    'maiden_over': 12,           # 12 points per maiden (calculated separately)
    'dot_ball': 0,               # Points per dot ball (for bowlers)
    'duck_penalty': -2,          # -2 for getting out on 0 (for batters who faced 5+ balls)
    '30_run_bonus': 4,           # +4 for scoring 30+ runs
    '50_run_bonus': 8,           # +8 for scoring 50+ runs (additional)
    '100_run_bonus': 16,         # +16 for scoring 100+ runs (additional)
    '3_wicket_bonus': 4,         # +4 for taking 3+ wickets
    '4_wicket_bonus': 8,         # +8 for taking 4+ wickets (additional)
    '5_wicket_bonus': 16,        # +16 for taking 5+ wickets (additional)
}

# Wicket types that give bonus to bowler
BOWLER_BONUS_WICKETS = ['bowled', 'lbw']


# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================

class DataPreprocessor:
    """Handles loading and preprocessing of IPL ball-by-ball data."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.player_match_data = None

    def load_data(self):
        """Load the CSV file and perform initial cleaning."""
        print("Loading data from:", self.file_path)
        self.raw_data = pd.read_csv(self.file_path)

        # Convert date column
        self.raw_data['start_date'] = pd.to_datetime(self.raw_data['start_date'])

        # Fill NaN values appropriately
        numeric_cols = ['runs_off_bat', 'extras', 'wides', 'noballs', 'byes', 'legbyes']
        for col in numeric_cols:
            if col in self.raw_data.columns:
                self.raw_data[col] = self.raw_data[col].fillna(0).astype(int)

        print(f"Loaded {len(self.raw_data):,} ball-by-ball records")
        print(f"Seasons: {self.raw_data['season'].min()} - {self.raw_data['season'].max()}")
        print(f"Unique matches: {self.raw_data['match_id'].nunique()}")

        return self.raw_data

    def calculate_batting_points(self):
        """Calculate fantasy points for batting performances."""
        df = self.raw_data.copy()

        # Group by match and player (as striker)
        batting_stats = df.groupby(['match_id', 'season', 'start_date', 'venue',
                                     'batting_team', 'bowling_team', 'striker']).agg({
            'runs_off_bat': 'sum',
            'ball': 'count',  # balls faced
        }).reset_index()

        batting_stats.columns = ['match_id', 'season', 'start_date', 'venue',
                                  'team', 'opponent', 'player', 'runs', 'balls_faced']

        # Count boundaries (4s and 6s)
        fours = df[df['runs_off_bat'] == 4].groupby(['match_id', 'striker']).size().reset_index(name='fours')
        sixes = df[df['runs_off_bat'] == 6].groupby(['match_id', 'striker']).size().reset_index(name='sixes')

        batting_stats = batting_stats.merge(fours, left_on=['match_id', 'player'],
                                             right_on=['match_id', 'striker'], how='left')
        batting_stats = batting_stats.merge(sixes, left_on=['match_id', 'player'],
                                             right_on=['match_id', 'striker'], how='left')

        batting_stats['fours'] = batting_stats['fours'].fillna(0).astype(int)
        batting_stats['sixes'] = batting_stats['sixes'].fillna(0).astype(int)

        # Remove duplicate columns
        batting_stats = batting_stats.drop(columns=['striker_x', 'striker_y'], errors='ignore')

        # Check for dismissals (ducks)
        dismissals = df[df['player_dismissed'].notna()][['match_id', 'player_dismissed']].drop_duplicates()
        dismissals['was_dismissed'] = True
        batting_stats = batting_stats.merge(dismissals, left_on=['match_id', 'player'],
                                             right_on=['match_id', 'player_dismissed'], how='left')
        batting_stats['was_dismissed'] = batting_stats['was_dismissed'].fillna(False)

        # Calculate batting fantasy points
        batting_stats['batting_points'] = (
            batting_stats['runs'] * POINTS_CONFIG['run'] +
            batting_stats['fours'] * POINTS_CONFIG['boundary_bonus'] +
            batting_stats['sixes'] * POINTS_CONFIG['six_bonus']
        )

        # Milestone bonuses
        batting_stats['batting_points'] += np.where(batting_stats['runs'] >= 30, POINTS_CONFIG['30_run_bonus'], 0)
        batting_stats['batting_points'] += np.where(batting_stats['runs'] >= 50, POINTS_CONFIG['50_run_bonus'], 0)
        batting_stats['batting_points'] += np.where(batting_stats['runs'] >= 100, POINTS_CONFIG['100_run_bonus'], 0)

        # Duck penalty (out on 0, faced at least 5 balls)
        batting_stats['batting_points'] += np.where(
            (batting_stats['runs'] == 0) & (batting_stats['was_dismissed']) & (batting_stats['balls_faced'] >= 5),
            POINTS_CONFIG['duck_penalty'], 0
        )

        return batting_stats[['match_id', 'season', 'start_date', 'venue', 'team',
                               'opponent', 'player', 'runs', 'balls_faced', 'fours',
                               'sixes', 'batting_points']]

    def calculate_bowling_points(self):
        """Calculate fantasy points for bowling performances."""
        df = self.raw_data.copy()

        # Calculate bowling stats
        bowling_stats = df.groupby(['match_id', 'season', 'start_date', 'venue',
                                     'bowling_team', 'batting_team', 'bowler']).agg({
            'ball': 'count',
            'runs_off_bat': 'sum',
            'extras': 'sum',
            'wides': 'sum',
            'noballs': 'sum',
        }).reset_index()

        bowling_stats.columns = ['match_id', 'season', 'start_date', 'venue',
                                  'team', 'opponent', 'player', 'balls_bowled',
                                  'runs_conceded', 'extras', 'wides', 'noballs']

        # Count wickets for each bowler
        wickets_df = df[df['wicket_type'].notna() &
                        ~df['wicket_type'].isin(['run out', 'retired hurt', 'obstructing the field'])]

        wickets = wickets_df.groupby(['match_id', 'bowler']).agg({
            'wicket_type': 'count'
        }).reset_index()
        wickets.columns = ['match_id', 'player', 'wickets']

        # Count LBW/Bowled wickets for bonus
        lbw_bowled = wickets_df[wickets_df['wicket_type'].isin(BOWLER_BONUS_WICKETS)]
        lbw_bowled_count = lbw_bowled.groupby(['match_id', 'bowler']).size().reset_index(name='lbw_bowled_wickets')

        bowling_stats = bowling_stats.merge(wickets, on=['match_id', 'player'], how='left')
        bowling_stats = bowling_stats.merge(lbw_bowled_count, left_on=['match_id', 'player'],
                                             right_on=['match_id', 'bowler'], how='left')

        bowling_stats['wickets'] = bowling_stats['wickets'].fillna(0).astype(int)
        bowling_stats['lbw_bowled_wickets'] = bowling_stats['lbw_bowled_wickets'].fillna(0).astype(int)

        # Calculate dot balls
        dot_balls = df[df['runs_off_bat'] == 0].groupby(['match_id', 'bowler']).size().reset_index(name='dot_balls')
        bowling_stats = bowling_stats.merge(dot_balls, left_on=['match_id', 'player'],
                                             right_on=['match_id', 'bowler'], how='left')
        bowling_stats['dot_balls'] = bowling_stats['dot_balls'].fillna(0).astype(int)

        # Drop duplicate columns
        bowling_stats = bowling_stats.drop(columns=['bowler_x', 'bowler_y', 'bowler'], errors='ignore')

        # Calculate bowling fantasy points
        bowling_stats['bowling_points'] = (
            bowling_stats['wickets'] * POINTS_CONFIG['wicket'] +
            bowling_stats['lbw_bowled_wickets'] * POINTS_CONFIG['lbw_bowled_bonus'] +
            bowling_stats['dot_balls'] * POINTS_CONFIG['dot_ball']
        )

        # Wicket milestone bonuses
        bowling_stats['bowling_points'] += np.where(bowling_stats['wickets'] >= 3, POINTS_CONFIG['3_wicket_bonus'], 0)
        bowling_stats['bowling_points'] += np.where(bowling_stats['wickets'] >= 4, POINTS_CONFIG['4_wicket_bonus'], 0)
        bowling_stats['bowling_points'] += np.where(bowling_stats['wickets'] >= 5, POINTS_CONFIG['5_wicket_bonus'], 0)

        return bowling_stats[['match_id', 'season', 'start_date', 'venue', 'team',
                               'opponent', 'player', 'balls_bowled', 'runs_conceded',
                               'wickets', 'dot_balls', 'bowling_points']]

    def aggregate_player_match_data(self):
        """Combine batting and bowling stats to get one row per player per match."""
        print("\nCalculating batting points...")
        batting = self.calculate_batting_points()

        print("Calculating bowling points...")
        bowling = self.calculate_bowling_points()

        # Merge batting and bowling stats
        print("Aggregating player match data...")

        # For players who only batted
        batting_only = batting.copy()
        batting_only['balls_bowled'] = 0
        batting_only['runs_conceded'] = 0
        batting_only['wickets'] = 0
        batting_only['dot_balls'] = 0
        batting_only['bowling_points'] = 0

        # For players who only bowled
        bowling_only = bowling.copy()
        bowling_only['runs'] = 0
        bowling_only['balls_faced'] = 0
        bowling_only['fours'] = 0
        bowling_only['sixes'] = 0
        bowling_only['batting_points'] = 0

        # Combine - merge on player and match
        combined = batting.merge(bowling[['match_id', 'player', 'balls_bowled',
                                           'runs_conceded', 'wickets', 'dot_balls', 'bowling_points']],
                                  on=['match_id', 'player'], how='outer')

        # Fill missing values
        for col in ['runs', 'balls_faced', 'fours', 'sixes', 'batting_points']:
            if col in combined.columns:
                combined[col] = combined[col].fillna(0)

        for col in ['balls_bowled', 'runs_conceded', 'wickets', 'dot_balls', 'bowling_points']:
            if col in combined.columns:
                combined[col] = combined[col].fillna(0)

        # Fill match metadata from non-null values
        for idx in combined[combined['season'].isna()].index:
            match_id = combined.loc[idx, 'match_id']
            match_data = self.raw_data[self.raw_data['match_id'] == match_id].iloc[0]
            combined.loc[idx, 'season'] = match_data['season']
            combined.loc[idx, 'start_date'] = match_data['start_date']
            combined.loc[idx, 'venue'] = match_data['venue']
            # For bowlers, team is bowling_team
            combined.loc[idx, 'team'] = match_data['bowling_team']
            combined.loc[idx, 'opponent'] = match_data['batting_team']

        # Calculate total fantasy points
        combined['Actual_Fantasy_Points'] = combined['batting_points'] + combined['bowling_points']

        # Convert numeric columns to int
        int_cols = ['runs', 'balls_faced', 'fours', 'sixes', 'balls_bowled',
                    'runs_conceded', 'wickets', 'dot_balls']
        for col in int_cols:
            combined[col] = combined[col].astype(int)

        # Sort by date and match
        combined = combined.sort_values(['start_date', 'match_id', 'player']).reset_index(drop=True)

        self.player_match_data = combined

        print(f"\nAggregated data: {len(combined):,} player-match records")
        print(f"Unique players: {combined['player'].nunique()}")
        print(f"Average fantasy points: {combined['Actual_Fantasy_Points'].mean():.2f}")

        return combined


# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """Creates predictive features from historical player performance data."""

    def __init__(self, player_match_data):
        self.data = player_match_data.copy()
        self.features_created = []

    def create_rolling_averages(self, windows=[3, 5, 10]):
        """
        Calculate rolling average fantasy points for each player.
        Uses shift(1) to prevent data leakage - only uses past games.
        """
        print("\nCreating rolling average features...")

        # Sort data by player and date
        self.data = self.data.sort_values(['player', 'start_date', 'match_id']).reset_index(drop=True)

        for window in windows:
            col_name = f'Avg_Points_Last_{window}_Matches'

            # Group by player and calculate rolling mean with shift
            self.data[col_name] = self.data.groupby('player')['Actual_Fantasy_Points'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )

            # Fill NaN for first matches (use overall player average as fallback)
            player_means = self.data.groupby('player')['Actual_Fantasy_Points'].transform('mean')
            self.data[col_name] = self.data[col_name].fillna(player_means)

            self.features_created.append(col_name)
            print(f"  - Created {col_name}")

        # Create recent form indicator (trend in last 3 vs last 10)
        if 'Avg_Points_Last_3_Matches' in self.data.columns and 'Avg_Points_Last_10_Matches' in self.data.columns:
            self.data['Recent_Form'] = (
                self.data['Avg_Points_Last_3_Matches'] - self.data['Avg_Points_Last_10_Matches']
            )
            self.features_created.append('Recent_Form')
            print("  - Created Recent_Form (trend indicator)")

        return self.data

    def create_venue_stats(self):
        """
        Calculate average fantasy points at each venue for each player.
        Uses only historical data (shift to prevent leakage).
        """
        print("\nCreating venue-specific features...")

        # Sort data
        self.data = self.data.sort_values(['player', 'venue', 'start_date']).reset_index(drop=True)

        # Calculate expanding mean by player-venue (shifted)
        self.data['Avg_Points_At_Venue'] = self.data.groupby(['player', 'venue'])['Actual_Fantasy_Points'].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # Fill NaN with player's overall average
        player_means = self.data.groupby('player')['Actual_Fantasy_Points'].transform('mean')
        self.data['Avg_Points_At_Venue'] = self.data['Avg_Points_At_Venue'].fillna(player_means)

        # Count matches at venue (for confidence weighting)
        self.data['Matches_At_Venue'] = self.data.groupby(['player', 'venue']).cumcount()

        self.features_created.extend(['Avg_Points_At_Venue', 'Matches_At_Venue'])
        print("  - Created Avg_Points_At_Venue")
        print("  - Created Matches_At_Venue")

        return self.data

    def create_opposition_stats(self):
        """
        Calculate average fantasy points against each opponent for each player.
        Uses only historical data (shift to prevent leakage).
        """
        print("\nCreating opposition-specific features...")

        # Sort data
        self.data = self.data.sort_values(['player', 'opponent', 'start_date']).reset_index(drop=True)

        # Calculate expanding mean by player-opponent (shifted)
        self.data['Avg_Points_Vs_Opponent'] = self.data.groupby(['player', 'opponent'])['Actual_Fantasy_Points'].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # Fill NaN with player's overall average
        player_means = self.data.groupby('player')['Actual_Fantasy_Points'].transform('mean')
        self.data['Avg_Points_Vs_Opponent'] = self.data['Avg_Points_Vs_Opponent'].fillna(player_means)

        # Count matches against opponent
        self.data['Matches_Vs_Opponent'] = self.data.groupby(['player', 'opponent']).cumcount()

        self.features_created.extend(['Avg_Points_Vs_Opponent', 'Matches_Vs_Opponent'])
        print("  - Created Avg_Points_Vs_Opponent")
        print("  - Created Matches_Vs_Opponent")

        return self.data

    def create_additional_features(self):
        """Create additional predictive features."""
        print("\nCreating additional features...")

        # Player career stats (expanding, shifted)
        self.data = self.data.sort_values(['player', 'start_date']).reset_index(drop=True)

        # Career average
        self.data['Career_Avg_Points'] = self.data.groupby('player')['Actual_Fantasy_Points'].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        player_means = self.data.groupby('player')['Actual_Fantasy_Points'].transform('mean')
        self.data['Career_Avg_Points'] = self.data['Career_Avg_Points'].fillna(player_means)

        # Career matches played
        self.data['Career_Matches'] = self.data.groupby('player').cumcount()

        # Season average (current season, shifted)
        self.data['Season_Avg_Points'] = self.data.groupby(['player', 'season'])['Actual_Fantasy_Points'].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        self.data['Season_Avg_Points'] = self.data['Season_Avg_Points'].fillna(self.data['Career_Avg_Points'])

        # Rolling batting stats
        self.data['Avg_Runs_Last_5'] = self.data.groupby('player')['runs'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )

        # Rolling bowling stats
        self.data['Avg_Wickets_Last_5'] = self.data.groupby('player')['wickets'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )

        # Fill NaN values
        self.data['Avg_Runs_Last_5'] = self.data['Avg_Runs_Last_5'].fillna(
            self.data.groupby('player')['runs'].transform('mean')
        )
        self.data['Avg_Wickets_Last_5'] = self.data['Avg_Wickets_Last_5'].fillna(
            self.data.groupby('player')['wickets'].transform('mean')
        )

        # Standard deviation of points (consistency measure)
        self.data['Points_Std'] = self.data.groupby('player')['Actual_Fantasy_Points'].transform(
            lambda x: x.shift(1).expanding().std()
        )
        self.data['Points_Std'] = self.data['Points_Std'].fillna(0)

        # Max points in career (ceiling indicator)
        self.data['Career_Max_Points'] = self.data.groupby('player')['Actual_Fantasy_Points'].transform(
            lambda x: x.shift(1).expanding().max()
        )
        self.data['Career_Max_Points'] = self.data['Career_Max_Points'].fillna(
            self.data.groupby('player')['Actual_Fantasy_Points'].transform('mean')
        )

        new_features = ['Career_Avg_Points', 'Career_Matches', 'Season_Avg_Points',
                        'Avg_Runs_Last_5', 'Avg_Wickets_Last_5', 'Points_Std', 'Career_Max_Points']
        self.features_created.extend(new_features)

        for feat in new_features:
            print(f"  - Created {feat}")

        return self.data

    def get_feature_columns(self):
        """Return list of all feature columns created."""
        return self.features_created

    def engineer_all_features(self):
        """Run all feature engineering steps."""
        self.create_rolling_averages([3, 5, 10])
        self.create_venue_stats()
        self.create_opposition_stats()
        self.create_additional_features()

        print(f"\n=== Feature Engineering Complete ===")
        print(f"Total features created: {len(self.features_created)}")

        return self.data


# =============================================================================
# 3. MODEL TRAINING
# =============================================================================

class Dream11Model:
    """Machine Learning model for predicting fantasy points."""

    def __init__(self, use_xgboost=True):
        self.model = None
        self.feature_columns = None
        self.use_xgboost = use_xgboost and XGBOOST_AVAILABLE
        self.feature_importance = None

    def prepare_data(self, data, feature_columns, train_seasons, test_seasons):
        """Split data into train and test sets based on seasons."""

        # Handle season ranges
        if isinstance(train_seasons, tuple):
            train_mask = (data['season'] >= train_seasons[0]) & (data['season'] <= train_seasons[1])
        else:
            train_mask = data['season'].isin(train_seasons)

        if isinstance(test_seasons, tuple):
            test_mask = (data['season'] >= test_seasons[0]) & (data['season'] <= test_seasons[1])
        else:
            test_mask = data['season'].isin(test_seasons)

        train_data = data[train_mask].copy()
        test_data = data[test_mask].copy()

        # Remove rows with NaN in features
        train_data = train_data.dropna(subset=feature_columns)
        test_data = test_data.dropna(subset=feature_columns)

        X_train = train_data[feature_columns]
        y_train = train_data['Actual_Fantasy_Points']
        X_test = test_data[feature_columns]
        y_test = test_data['Actual_Fantasy_Points']

        self.feature_columns = feature_columns

        return X_train, X_test, y_train, y_test, train_data, test_data

    def train(self, X_train, y_train):
        """Train the prediction model."""
        print("\n=== Training Model ===")

        if self.use_xgboost:
            print("Using XGBoost Regressor...")
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        else:
            print("Using Random Forest Regressor...")
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

        print(f"Training on {len(X_train):,} samples...")
        self.model.fit(X_train, y_train)

        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)

        print("Training complete!")
        return self.model

    def evaluate(self, X_test, y_test, test_data=None):
        """Evaluate model performance."""
        print("\n=== Model Evaluation ===")

        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Absolute Error (MAE): {mae:.2f} points")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} points")
        print(f"R-squared Score: {r2:.4f}")

        # Analyze predictions by points range
        if test_data is not None:
            test_data = test_data.copy()
            test_data['Predicted_Points'] = y_pred
            test_data['Prediction_Error'] = abs(y_pred - y_test)

            print("\nPerformance by actual points range:")
            bins = [0, 10, 25, 50, 100, 500]
            labels = ['0-10', '10-25', '25-50', '50-100', '100+']
            test_data['Points_Range'] = pd.cut(y_test, bins=bins, labels=labels)

            range_stats = test_data.groupby('Points_Range', observed=True).agg({
                'Prediction_Error': ['mean', 'count']
            }).round(2)
            print(range_stats)

        return {'mae': mae, 'rmse': rmse, 'r2': r2}

    def predict(self, X):
        """Make predictions for new data."""
        return self.model.predict(X)

    def save_model(self, filepath):
        """Save the trained model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance
            }, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_columns = data['feature_columns']
            self.feature_importance = data.get('feature_importance')
        print(f"Model loaded from {filepath}")


# =============================================================================
# 4. DREAM11 TEAM PREDICTOR
# =============================================================================

class Dream11Predictor:
    """Predicts optimal Dream11 team for an upcoming match."""

    def __init__(self, model, player_data, feature_columns):
        self.model = model
        self.player_data = player_data
        self.feature_columns = feature_columns
        self.latest_player_stats = self._compute_latest_stats()

    def _compute_latest_stats(self):
        """Compute the most recent stats for each player."""
        # Get the latest record for each player
        latest = self.player_data.sort_values('start_date').groupby('player').last().reset_index()
        return latest

    def get_team_players(self, team_name):
        """Get list of players who have played for a team recently."""
        # Get players from the most recent season
        max_season = self.player_data['season'].max()
        recent_data = self.player_data[self.player_data['season'] == max_season]

        # Flexible team matching
        team_lower = team_name.lower()
        team_matches = recent_data[
            recent_data['team'].str.lower().str.contains(team_lower, na=False)
        ]

        if len(team_matches) == 0:
            # Try partial matching
            print(f"Warning: Team '{team_name}' not found. Available teams:")
            print(recent_data['team'].unique())
            return []

        players = team_matches['player'].unique().tolist()
        return players

    def prepare_prediction_features(self, player, opponent, venue):
        """Prepare features for a single player prediction."""
        # Get player's latest stats
        player_stats = self.latest_player_stats[
            self.latest_player_stats['player'] == player
        ]

        if len(player_stats) == 0:
            return None

        player_stats = player_stats.iloc[0].copy()

        # Update venue-specific stats if available
        venue_data = self.player_data[
            (self.player_data['player'] == player) &
            (self.player_data['venue'].str.contains(venue, case=False, na=False))
        ]
        if len(venue_data) > 0:
            player_stats['Avg_Points_At_Venue'] = venue_data['Actual_Fantasy_Points'].mean()
            player_stats['Matches_At_Venue'] = len(venue_data)

        # Update opponent-specific stats if available
        opp_data = self.player_data[
            (self.player_data['player'] == player) &
            (self.player_data['opponent'].str.lower().str.contains(opponent.lower(), na=False))
        ]
        if len(opp_data) > 0:
            player_stats['Avg_Points_Vs_Opponent'] = opp_data['Actual_Fantasy_Points'].mean()
            player_stats['Matches_Vs_Opponent'] = len(opp_data)

        # Extract only feature columns
        features = {}
        for col in self.feature_columns:
            if col in player_stats.index:
                features[col] = player_stats[col]
            else:
                features[col] = 0

        return pd.DataFrame([features])

    def predict_dream11_team(self, team_a, team_b, venue, top_n=11):
        """
        Predict the best Dream11 team for a match between team_a and team_b.

        Parameters:
        -----------
        team_a : str
            Name of the first team (e.g., "Mumbai Indians")
        team_b : str
            Name of the second team (e.g., "Chennai Super Kings")
        venue : str
            Venue of the match (e.g., "Wankhede Stadium")
        top_n : int
            Number of top players to return (default: 11)

        Returns:
        --------
        DataFrame with predicted fantasy points for all players, sorted by prediction
        """
        print(f"\n{'='*60}")
        print(f"DREAM11 PREDICTION: {team_a} vs {team_b}")
        print(f"Venue: {venue}")
        print(f"{'='*60}")

        # Get players from both teams
        team_a_players = self.get_team_players(team_a)
        team_b_players = self.get_team_players(team_b)

        print(f"\n{team_a}: {len(team_a_players)} players found")
        print(f"{team_b}: {len(team_b_players)} players found")

        all_predictions = []

        # Predict for Team A players
        for player in team_a_players:
            features = self.prepare_prediction_features(player, team_b, venue)
            if features is not None:
                prediction = self.model.predict(features)[0]
                all_predictions.append({
                    'Player': player,
                    'Team': team_a,
                    'Predicted_Points': round(prediction, 2),
                    'Career_Avg': round(features['Career_Avg_Points'].values[0], 2) if 'Career_Avg_Points' in features.columns else 0,
                    'Recent_Form': round(features['Avg_Points_Last_5_Matches'].values[0], 2) if 'Avg_Points_Last_5_Matches' in features.columns else 0
                })

        # Predict for Team B players
        for player in team_b_players:
            features = self.prepare_prediction_features(player, team_a, venue)
            if features is not None:
                prediction = self.model.predict(features)[0]
                all_predictions.append({
                    'Player': player,
                    'Team': team_b,
                    'Predicted_Points': round(prediction, 2),
                    'Career_Avg': round(features['Career_Avg_Points'].values[0], 2) if 'Career_Avg_Points' in features.columns else 0,
                    'Recent_Form': round(features['Avg_Points_Last_5_Matches'].values[0], 2) if 'Avg_Points_Last_5_Matches' in features.columns else 0
                })

        # Create DataFrame and sort
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df = predictions_df.sort_values('Predicted_Points', ascending=False).reset_index(drop=True)

        # Display top picks
        print(f"\n{'='*60}")
        print(f"TOP {top_n} DREAM11 PICKS")
        print(f"{'='*60}")

        top_picks = predictions_df.head(top_n)
        for idx, row in top_picks.iterrows():
            print(f"{idx+1:2}. {row['Player']:<25} ({row['Team']:<20}) - {row['Predicted_Points']:6.2f} pts")

        # Captain and Vice-Captain recommendations
        print(f"\n{'='*60}")
        print("CAPTAIN & VICE-CAPTAIN RECOMMENDATIONS")
        print(f"{'='*60}")
        if len(predictions_df) >= 2:
            print(f"CAPTAIN (2x):       {predictions_df.iloc[0]['Player']} ({predictions_df.iloc[0]['Predicted_Points']:.2f} pts)")
            print(f"VICE-CAPTAIN (1.5x): {predictions_df.iloc[1]['Player']} ({predictions_df.iloc[1]['Predicted_Points']:.2f} pts)")

        # Team balance check
        print(f"\n{'='*60}")
        print("TEAM COMPOSITION (Top 11)")
        print(f"{'='*60}")
        team_counts = top_picks['Team'].value_counts()
        for team, count in team_counts.items():
            print(f"  {team}: {count} players")

        return predictions_df


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""

    # Configuration
    DATA_FILE = "IPL_ball_by_ball_updated.csv"
    MODEL_FILE = "dream11_model.pkl"

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_FILE)
    model_path = os.path.join(script_dir, MODEL_FILE)

    print("="*60)
    print("DREAM11 IPL FANTASY POINTS PREDICTION ENGINE")
    print("="*60)

    # ==========================================================================
    # STEP 1: Data Preprocessing
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)

    preprocessor = DataPreprocessor(data_path)
    preprocessor.load_data()
    player_match_data = preprocessor.aggregate_player_match_data()

    # ==========================================================================
    # STEP 2: Feature Engineering
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)

    feature_engineer = FeatureEngineer(player_match_data)
    featured_data = feature_engineer.engineer_all_features()
    feature_columns = feature_engineer.get_feature_columns()

    print(f"\nFeature columns: {feature_columns}")

    # ==========================================================================
    # STEP 3: Model Training
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING")
    print("="*60)

    model = Dream11Model(use_xgboost=True)

    # Determine train/test split based on available data
    max_season = featured_data['season'].max()
    test_season = max_season  # Use most recent season for testing

    print(f"Data available up to season: {max_season}")
    print(f"Training on: 2008-{max_season - 1}, Testing on: {max_season}")

    X_train, X_test, y_train, y_test, train_data, test_data = model.prepare_data(
        featured_data,
        feature_columns,
        train_seasons=(2008, max_season - 1),
        test_seasons=(max_season, max_season)
    )

    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")

    # Train the model
    model.train(X_train, y_train)

    # Evaluate
    metrics = model.evaluate(X_test, y_test, test_data)

    # Feature Importance
    if model.feature_importance is not None:
        print("\nTop 10 Feature Importance:")
        print(model.feature_importance.head(10).to_string(index=False))

    # Save the model
    model.save_model(model_path)

    # ==========================================================================
    # STEP 4: Prediction Demo
    # ==========================================================================
    print("\n" + "="*60)
    print("STEP 4: DREAM11 PREDICTION DEMO")
    print("="*60)

    predictor = Dream11Predictor(model.model, featured_data, feature_columns)

    # Example prediction
    predictions = predictor.predict_dream11_team(
        team_a="Mumbai Indians",
        team_b="Chennai Super Kings",
        venue="Wankhede Stadium"
    )

    # Save predictions to CSV
    predictions_file = os.path.join(script_dir, "dream11_predictions.csv")
    predictions.to_csv(predictions_file, index=False)
    print(f"\nPredictions saved to: {predictions_file}")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)

    return model, predictor, featured_data


def predict_dream11_team(team_a, team_b, venue, model_path="dream11_model.pkl", data_path="IPL_ball_by_ball_updated.csv"):
    """
    Standalone function to predict Dream11 team for a match.

    Parameters:
    -----------
    team_a : str
        Name of the first team
    team_b : str
        Name of the second team
    venue : str
        Venue of the match
    model_path : str
        Path to the saved model file
    data_path : str
        Path to the IPL data CSV file

    Returns:
    --------
    DataFrame with player predictions
    """
    # Get script directory for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isabs(model_path):
        model_path = os.path.join(script_dir, model_path)
    if not os.path.isabs(data_path):
        data_path = os.path.join(script_dir, data_path)

    # Load the model
    model = Dream11Model()
    model.load_model(model_path)

    # Load and process data
    preprocessor = DataPreprocessor(data_path)
    preprocessor.load_data()
    player_match_data = preprocessor.aggregate_player_match_data()

    # Feature engineering
    feature_engineer = FeatureEngineer(player_match_data)
    featured_data = feature_engineer.engineer_all_features()

    # Create predictor and predict
    predictor = Dream11Predictor(model.model, featured_data, model.feature_columns)
    predictions = predictor.predict_dream11_team(team_a, team_b, venue)

    return predictions


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run the full pipeline
    model, predictor, data = main()

    # Example: Make additional predictions
    print("\n" + "="*60)
    print("ADDITIONAL PREDICTIONS")
    print("="*60)

    # You can call the predictor for other matches:
    # predictor.predict_dream11_team("Royal Challengers Bangalore", "Kolkata Knight Riders", "M Chinnaswamy Stadium")
