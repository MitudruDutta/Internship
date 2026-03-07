# Week 1 Task 2: Nova Mart Visual Analysis and Presentation Support

This task turns the Nova Mart campaign analysis into charts and presentation-ready insights.

## Objective

The goal is to answer the second set of client requests with visual outputs that are easier to present to stakeholders. The notebook focuses on comparisons, trends, and campaign effectiveness across cities, categories, and promo types.

## Visual analyses included

`task2.ipynb` produces charts and summaries for:

- store count by city
- Sankranti post-promo quantity contribution by category
- relationship between post-promo price and quantity sold
- quantity sold before promotion across categories
- city-wise `ISU%`
- Hyderabad promo comparison using `IR%` and `ISU%`
- Bengaluru revenue before versus after promotion by category

## Files in this directory

- `task2.ipynb`: main visualization notebook
- `datasets/`: source tables used in the joins and calculations
- `output/`: exported chart images from the notebook
- `meta_data.txt`: data dictionary
- `client_requests.pdf`: task brief
- `presentation_outline.md`: presentation structure helper
- `presentation_prompt.txt`: presentation-generation helper prompt
- `video_script.md`: speaking script for the presentation

## How to run

1. Install dependencies from the repository root with `pip install -r requirements.txt`.
2. Start Jupyter with `jupyter notebook`.
3. Open `task2.ipynb` and run all cells.
4. Review generated charts in `output/`.

## Notes

- This task is the presentation layer for the Nova Mart analysis, so the outputs are intended for slide-making and client communication.
- The helper prompt and video script are local working files and may not be tracked in the remote repository.
