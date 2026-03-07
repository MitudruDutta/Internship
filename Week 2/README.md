# Week 2: Statistical Analysis and EV SQL Debugging

Week 2 contains two separate tasks. The first is a Python notebook for statistics-based client questions. The second is a SQL debugging task for EV sales analysis.

## Task 1: statistical analysis in Python

`Task 1/task1.ipynb` answers a sequence of client-request questions on e-commerce customer behavior. The notebook covers:

- age range
- mean, median, and mode of amount spent
- variance and standard deviation of time spent on site
- interquartile range of amount spent
- correlation between purchases and time spent
- z-score calculation
- skewness
- probability questions
- confidence interval estimation
- conditional probability for high cross-sell conversion

Supporting local files:

- `Task 1/datasets/ecommerce_data.csv`
- `Task 1/meta_data.txt`
- `Task 1/client_requests.pdf`

## Task 2: EV sales SQL debugging

`Task 2/sql/fixed_sql_queries.sql` contains corrected versions of the broken intern SQL queries. The task works on the EV sales database dump and covers:

- maker counts by vehicle category
- top makers by fiscal year
- monthly average vehicle sales
- state penetration rate ranking
- highest and lowest sales states
- peak and low season months
- CAGR for top makers
- penetration category classification by state

Supporting local files:

- `Task 2/sql/ev_sales_db.sql`
- `Task 2/sql_queries.docx`
- `Task 2/meta_data.txt`

## How to run

For Task 1:

1. Install dependencies from the repository root with `pip install -r requirements.txt`.
2. Start Jupyter with `jupyter notebook`.
3. Open `Task 1/task1.ipynb`.

For Task 2:

1. Load `Task 2/sql/ev_sales_db.sql` into MySQL Workbench or another compatible SQL engine.
2. Review the problem statements in `Task 2/sql_queries.docx`.
3. Run the corrected queries in `Task 2/sql/fixed_sql_queries.sql`.
