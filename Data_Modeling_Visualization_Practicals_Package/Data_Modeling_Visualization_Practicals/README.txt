COMPUTER LABORATORY-I: DATA MODELING & VISUALIZATION (417525)

Contains 7 Python scripts, 7 Jupyter notebooks (one code cell each), and datasets.
1 Sales data in CSV, Excel and JSON
2 Weather API with offline sample fallback
3 Telecom customer churn cleaning
4 Real estate data wrangling
5 AQI visualization
6 Retail sales aggregation
7 Stock market time-series analysis

Install:
pip install pandas numpy matplotlib openpyxl requests

Open this folder in VS Code. Run notebooks in notebooks/ or scripts in practicals/.
For scripts, open terminal in practicals/ and run: python 01_Sales_Multiple_Formats.py

Weather API: To fetch live data, set environment variable OPENWEATHER_API_KEY.
Without it, the practical automatically uses datasets/weather_sample.csv.

DATA NOTE: All supplied datasets are generated educational/demo data, not official college,
company, Kaggle, or real-time market datasets. The weather live API is used only when
you configure your own API key. The manual says Any 6 assignments; this package includes all 7.
