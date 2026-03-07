# Data Science and Machine Learning Internship Portfolio

This repository collects the work completed across the internship in three business contexts:

- `Week 1`: Nova Mart FMCG promotion analysis using Python, Pandas, Matplotlib, and Seaborn.
- `Week 2`: statistical analysis in Python and EV market SQL query debugging.
- `Week 3 and 4`: the CodeX beverage pricing project, covering cleaning, feature engineering, modeling, MLflow tracking, and deployment.

Each week solves a different client-style problem, so the repository is better read as a set of project deliverables than as a single application.

## Projects

### Week 1: Nova Mart promotion analysis

The first week focuses on promotional effectiveness for an FMCG client. The work answers operational questions such as store coverage by city, campaign lift during Diwali and Sankranti, category contribution, and promo-level revenue or unit growth.

Core business metrics used throughout the analysis:

- `IR%`: incremental revenue percentage
- `ISU%`: incremental sold units percentage

Primary files:

- [Week 1/README.md](Week%201/README.md)
- `Week 1/Task 1/task1.ipynb`
- `Week 1/Task 2/task2.ipynb`

### Week 2: statistics and SQL debugging

The second week combines two different tasks:

- a Python notebook that answers descriptive and inferential statistics questions on e-commerce customer behavior
- a SQL debugging exercise that fixes broken EV market queries and validates market metrics such as penetration rate and CAGR

Primary files:

- [Week 2/README.md](Week%202/README.md)
- `Week 2/Task 1/task1.ipynb`
- `Week 2/Task 2/sql/fixed_sql_queries.sql`

### Week 3 and 4: CodeX beverage pricing

The final phase builds an end-to-end machine learning solution for beverage price-range prediction. It includes survey cleaning, engineered features, model comparison, MLflow logging, a serialized LightGBM model artifact, a FastAPI inference service, and a Streamlit demo app.

Primary files:

- [Week 3 and 4/README.md](Week%203%20and%204/README.md)
- [Week 3 and 4/CodeX Project/README.md](Week%203%20and%204/CodeX%20Project/README.md)

## How to work with this repository

1. Create a Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Open the week you want to review and start from that directory's `README.md`.
4. Run notebooks in Jupyter for analysis tasks.
5. Run `uvicorn` and `streamlit` only for the CodeX deployment work in Week 3 and 4.

## Notes

- Some client instruction PDFs, presentation helpers, and generated assets are intentionally excluded from version control or kept local only.
- The repository contains both notebook-based analysis and deployable application code, so commands differ by week.
