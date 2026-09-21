# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Save the matching validation predictions to `output/validation_predictions.csv`.
4. Save December predictions to `output/december_predictions.csv`; keep `data/december_chart_inputs.csv` unchanged.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions output/validation_predictions.csv --december-predictions output/december_predictions.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `output/validation_predictions.csv`
- `output/december_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link