ML PRACTICALS - EXTERNAL EXAM PACKAGE
======================================

This package contains 7 separate Python scripts and 7 Jupyter notebooks.
Each notebook has ONE code cell. The scripts can also be run directly.

FOLDER STRUCTURE
- datasets/       CSV files used by the practicals
- practicals/     one .py file per practical
- notebooks/      one .ipynb file per practical (one code cell each)

SETUP
1. Extract this ZIP.
2. Open the extracted ML_Practicals_External_Package folder in VS Code.
3. Open Terminal and run:
   pip install pandas numpy matplotlib scikit-learn
4. Open notebooks/01_PCA_Wine.ipynb and select your Python kernel.
5. Run the single cell using Shift+Enter.
6. Repeat for each practical.

IMPORTANT DATASET NOTES
1. wine.csv: real Wine recognition dataset distributed by scikit-learn.
2. uber_fares_demo.csv: generated demonstration dataset with distance, passenger count,
   pickup hour and fare. It is NOT the full original Kaggle Uber Fares dataset cited
   by the manual. Use the full Kaggle file if your examiner specifically requires it.
3. handwritten_digits.csv: real handwritten digits dataset distributed by scikit-learn.
4. kmeans_points.csv: small 2D point dataset for demonstrating K-Means and centroids.
5. car_safety_demo.csv: generated teaching/demo data for Random Forest car safety.
   It is NOT the original Kaggle dataset referenced in the manual.
6. maze.csv: small maze grid; Reinforcement Learning does not require a conventional
   labelled CSV dataset, but this package stores the maze layout in CSV for convenience.
7. emotion_text_demo.csv: small demonstration text/emotion dataset. The supplied manual
   gives the experiment aim but does not specify a dataset or full implementation.

The manual's practicals are PCA, Regression Analysis, SVM digit classification,
K-Means clustering, Random Forest car safety, Reinforcement Learning and NLP emotion
detection. Where the manual gives an external dataset link, this package labels any
generated substitute clearly rather than presenting it as the original dataset.

If a notebook cannot find the CSV, make sure you opened VS Code at the extracted
package's main folder, not inside the notebooks folder.
