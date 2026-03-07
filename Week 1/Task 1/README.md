# Week 1 Task 1: Nova Mart Client Request Analysis

This task answers the direct business questions provided by Nova Mart using the campaign event data and the supporting dimension tables.

## Objective

The goal is to clean the raw event data, join the required tables, and compute exact answers for the client requests. The work focuses on operational and campaign-level questions rather than presentation design.

## Questions answered

`task1.ipynb` covers:

- duplicate event handling
- cities with more than 5 stores
- median imputation for missing pre-promo quantities
- lowest-priced product category before promotion
- total quantity sold after `BOGOF` during `Diwali`
- top-performing store during `Diwali`
- campaign lift comparison between `Sankranti` and `Diwali`
- highest `IR%` product during `Sankranti`
- lowest `ISU%` store in `Visakhapatnam` during `Diwali`
- promo types with negative `IR%` and `ISU%`

## Files in this directory

- `task1.ipynb`: main analysis notebook
- `datasets/`: campaign fact and dimension tables
- `meta_data.txt`: column definitions and dataset context
- `client_requests.pdf`: original client questions

## How to run

1. Install dependencies from the repository root with `pip install -r requirements.txt`.
2. Start Jupyter with `jupyter notebook`.
3. Open `task1.ipynb` and run the cells in order.

## Notes

- `IR%` and `ISU%` are calculated inside the notebook for the relevant campaign slices.
- Some local source files may be ignored from Git for repository cleanup, but the notebook is written to work with the provided task assets.
