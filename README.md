# 🎓 Student Marks Analytics Dashboard

Interactive Streamlit + Plotly dashboard for visualizing student marks (percentages) across subjects.
Data: `data/student_marks_dataset.csv` — 500 synthetic students, 6 subjects (generated for demonstration, not real records).

## Run in VS Code

1. Install Python 3.10+ and open this folder in VS Code (**File → Open Folder**).
2. Open a terminal (**Terminal → New Terminal**) and run:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

3. Your browser opens at <http://localhost:8501>.

## Pages

| Page | Charts |
|---|---|
| 🏠 Overview | Average per subject (bar), percentage histogram, grade donut, pass/fail by section, top/bottom students |
| 📚 Subject Analysis | Average + pass-rate combo, box plots, single-subject histogram, grouped bars by section/gender/grade, section × subject heatmap |
| 🧑‍🎓 Student Profile | Radar and bar chart of one student vs the class average, rank, strengths |
| ⚖️ Compare | Radar of groups or of 2–6 chosen students |
| 🔗 Correlations | Correlation heatmap, scatter explorer with trend line |
| 🧾 Data Explorer | Search, sort, summary statistics, **⬇️ Download Filtered Data** |

Sidebar filters: section, gender, result, percentage range, attendance. Advanced controls: pass mark (default 35).

## Use your own data

Replace `data/student_marks_dataset.csv`. Required columns: `Student_ID`, `Student_Name`, `Gender`, `Section`
and one numeric column per subject (marks out of 100). `Attendance_%` and `Study_Hours_Per_Day` are optional.
Subjects are detected automatically; Total, Percentage, Result and Grade are calculated by the app.

## Structure

```text
student-marks-dashboard/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── student_marks_dataset.csv
```
