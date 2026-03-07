# Week 2 Task 2: EV Sales SQL Query Debugging

This task fixes incorrect SQL written by previous interns and produces correct answers from the EV sales database.

## Objective

The focus is not schema design or dashboarding. The focus is query correction, metric validation, and accurate result generation from the provided relational data.

## SQL topics covered

The corrected query set in `sql/fixed_sql_queries.sql` covers:

- unique 2-wheeler makers
- top 3 makers in fiscal years 2023 and 2024
- average monthly total vehicle sales in fiscal year 2024
- top states by penetration rate for 2-wheelers and 4-wheelers
- highest and lowest total vehicle sales states in fiscal year 2023
- peak and low EV sales months from 2022 to 2024
- CAGR for top 2-wheeler makers from 2022 to 2024
- state penetration category classification in fiscal year 2024

## Files in this directory

- `sql/ev_sales_db.sql`: database dump to import into MySQL
- `sql/fixed_sql_queries.sql`: corrected query file
- `sql_queries.docx`: original broken query prompts
- `meta_data.txt`: schema and data description

## How to run

1. Import `sql/ev_sales_db.sql` into MySQL Workbench.
2. Review `sql_queries.docx` for the original problem statements.
3. Execute the queries from `sql/fixed_sql_queries.sql`.

## Notes

- The corrected SQL is written against the provided `ev_sales_db` schema.
- The results depend on the fiscal year mapping in `dim_date`, so that table must be imported correctly before testing the queries.
