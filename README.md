<p align="center">
  <img src="https://img.shields.io/badge/IPL-Fantasy%20Predictor-orange?style=for-the-badge&logo=cricket" alt="IPL Fantasy"/>
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-1.0+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🏏 IPL Oracle - Dream11 Predictor</h1>

<p align="center">
  <strong>Build winning fantasy cricket teams using Machine Learning predictions</strong>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 🎯 Overview

**IPL Oracle** is an intelligent fantasy cricket prediction system that leverages machine learning to help you build the optimal Dream11 team. Simply select two competing teams, and the system will analyze player statistics, recent form, and career averages to recommend the best XI with Captain and Vice-Captain picks.

> ⚡ **Stop guessing. Start winning.**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **ML-Powered Predictions** | Advanced algorithms analyze player performance metrics |
| 👑 **Smart C/VC Selection** | Automatically identifies the safest Captain and high-reward Vice-Captain |
| 📊 **Risk Analysis** | Highlights high-profile players predicted to underperform |
| 🎨 **Beautiful UI** | Clean, intuitive Streamlit dashboard with color-coded cards |
| ⚡ **Real-time Results** | Instant predictions with cached data loading |
| 📈 **Comprehensive Stats** | Career averages, recent form, and predicted points |

---

## 🖥️ Demo

### Main Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│  🏏 IPL Fantasy Cricket Predictor                           │
│  Build your winning Dream11 team using ML predictions       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   🔵 Chennai Super Kings  ⚔️ VS  🔴 Mumbai Indians          │
│                                                             │
│   ┌─────────────────┐    ┌─────────────────┐               │
│   │  👑 CAPTAIN     │    │  🥈 VICE-CAPTAIN│               │
│   │  JO Holder      │    │  RG Sharma      │               │
│   │  CSK            │    │  MI             │               │
│   │  119.2 pts (2x) │    │  84.5 pts (1.5x)│               │
│   └─────────────────┘    └─────────────────┘               │
│                                                             │
│   📋 Complete Squad                                         │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ Player          │ Team │ Predicted │ Career │ Form  │  │
│   ├─────────────────────────────────────────────────────┤  │
│   │ JO Holder       │ CSK  │ 59.6      │ 38.4   │ 5.0   │  │
│   │ RG Sharma       │ MI   │ 56.3      │ 35.7   │ 36.8  │  │
│   │ ...             │ ...  │ ...       │ ...    │ ...   │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   📈 Total Predicted Points: 487.3                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ipl-oracle.git
   cd ipl-oracle
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   ```
   http://localhost:8501
   ```

---

## 📦 Dependencies

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

Create a `requirements.txt` file:
```bash
pip freeze > requirements.txt
```

---

## 📖 Usage

### Step 1: Select Teams
Use the sidebar dropdowns to choose:
- **Team A** - First competing team
- **Team B** - Second competing team

### Step 2: Generate Predictions
Click the **"🎯 Predict Best XI"** button

### Step 3: Review Your Dream Team
The app displays:
- 👑 **Captain** - Highest predicted scorer (2x points)
- 🥈 **Vice-Captain** - Second highest scorer (1.5x points)
- 📋 **Complete Squad** - Top 11 players with stats
- ⚠️ **Risk Analysis** - Players predicted to underperform

---

## 🧠 How It Works

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Historical     │     │   ML Pipeline    │     │   Dream11        │
│   Player Data    │ ──▶ │   Processing     │ ──▶ │   Predictions    │
└──────────────────┘     └──────────────────┘     └──────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
   • Career Stats           • Feature Eng.          • Best XI
   • Recent Form            • Model Training        • C/VC Picks
   • Venue History          • Prediction            • Risk Analysis
```

### Prediction Features

| Feature | Description |
|---------|-------------|
| `Predicted_Points` | ML model's projected fantasy points |
| `Career_Avg` | Player's historical average performance |
| `Recent_Form` | Performance in last 5 matches |

### Selection Algorithm

1. **Filter** players from selected teams
2. **Rank** by predicted points
3. **Select** top 11 players
4. **Assign** Captain (highest) & Vice-Captain (2nd highest)
5. **Calculate** effective points with multipliers

---

## 📁 Project Structure

```
ipl-oracle/
│
├── app.py                      # Main Streamlit application
├── dream11_predictions.csv     # Player predictions dataset
├── IPL_ball_by_ball_updated.csv # Raw match data
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
├── models/                     # (Optional) Saved ML models
│   └── random_forest.pkl
│
└── notebooks/                  # (Optional) Jupyter notebooks
    └── data_analysis.ipynb
```

---

## 📊 Dataset Schema

### `dream11_predictions.csv`

| Column | Type | Description |
|--------|------|-------------|
| `Player` | string | Player name |
| `Team` | string | IPL franchise |
| `Predicted_Points` | float | ML predicted fantasy points |
| `Career_Avg` | float | Historical average points |
| `Recent_Form` | float | Last 5 matches average |

---

## 🎨 Screenshots

<details>
<summary>Click to expand screenshots</summary>

### Home Screen
> Select teams from the sidebar

### Prediction Results
> View your optimal Dream11 team

### Risk Analysis
> Identify underperforming stars

</details>

---

## 🛠️ Configuration

### Customizing the App

Edit `app.py` to modify:

```python
# Change data source
DATA_FILE = "your_predictions.csv"

# Adjust team size
dream_team = select_dream_team(match_players, top_n=11)

# Modify point multipliers
captain_multiplier = 2.0
vc_multiplier = 1.5
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit** your changes
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open** a Pull Request

### Ideas for Contribution

- [ ] Add player role filtering (WK, BAT, ALL, BOWL)
- [ ] Implement venue-specific predictions
- [ ] Add head-to-head statistics
- [ ] Create mobile-responsive design
- [ ] Add export to PDF/Excel feature

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 IPL Oracle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
```

---

## ⚠️ Disclaimer

> **This tool is for educational and entertainment purposes only.**
>
> Fantasy cricket involves risk. Past performance does not guarantee future results.
> The predictions are based on statistical models and should not be considered as
> financial or betting advice. Please play responsibly.

---

## 🙏 Acknowledgments

- **IPL** for the incredible cricket tournament
- **Dream11** for revolutionizing fantasy sports
- **Streamlit** for the amazing web framework
- **Scikit-learn** for ML capabilities

---

<p align="center">
  <strong>Made with ❤️ for Cricket Fans</strong>
</p>

<p align="center">
  <a href="https://github.com/yourusername/ipl-oracle/stargazers">⭐ Star this repo</a> •
  <a href="https://github.com/yourusername/ipl-oracle/issues">🐛 Report Bug</a> •
  <a href="https://github.com/yourusername/ipl-oracle/issues">💡 Request Feature</a>
</p>

---

<p align="center">
  <img src="https://img.shields.io/github/stars/yourusername/ipl-oracle?style=social" alt="Stars"/>
  <img src="https://img.shields.io/github/forks/yourusername/ipl-oracle?style=social" alt="Forks"/>
  <img src="https://img.shields.io/github/watchers/yourusername/ipl-oracle?style=social" alt="Watchers"/>
</p>
"# Dream11Oracle" 
