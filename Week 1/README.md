# Week 1: Nova Mart FMCG Promotion Analysis

This week focuses on promotional campaign analysis for Nova Mart. The objective is to answer client requests with clean calculations first, then convert those results into chart-based business communication.

## Business context

Nova Mart runs promotional campaigns across cities, stores, and product categories. The analysis is built to answer questions such as:

- which cities have stronger store coverage
- which categories contributed most during campaign periods
- which promotion types increased units versus revenue
- where campaign performance underperformed and needs attention

Two core metrics are used repeatedly:

- `IR% = ((revenue_after - revenue_before) / revenue_before) * 100`
- `ISU% = ((units_after - units_before) / units_before) * 100`

## Task 1

`Task 1/task1.ipynb` answers the direct client-request questions from the dataset. The notebook covers:

- duplicate removal and missing-value treatment
- city-wise store coverage
- category pricing comparisons
- campaign-specific quantity sold analysis
- promo and store performance during Diwali and Sankranti
- `IR%` and `ISU%` calculations for products, stores, and promo types

Supporting local files:

- `Task 1/datasets/`
- `Task 1/meta_data.txt`
- `Task 1/client_requests.pdf`

## Task 2

`Task 2/task2.ipynb` converts the campaign analysis into presentation-ready visuals. It includes charts for:

- number of stores by city
- category contribution during Sankranti
- price versus quantity relationship
- quantity distribution by category
- city-wise `ISU%`
- Hyderabad promo trade-off between `IR%` and `ISU%`
- Bengaluru category revenue before and after promotion

Exported chart files are stored in `Task 2/output/`.

Local presentation helper files are also present in `Task 2/`:

- `presentation_outline.md`
- `presentation_prompt.txt`
- `video_script.md`

## How to run

1. Install dependencies from the repository root with `pip install -r requirements.txt`.
2. Open Jupyter with `jupyter notebook`.
3. Run `Task 1/task1.ipynb` for the direct answers.
4. Run `Task 2/task2.ipynb` for the chart outputs and presentation material.
