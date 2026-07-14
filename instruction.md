Business context:
A marketing experiment is running across four acquisition channels. Each row in the dataset represents one user and records the acquisition channel, a treatment assignment, and whether the user converted during the experiment.

The experiment should have been randomized 50/50 between treatment and control across all users. The dataset is located at `/app/data/ab_test_data.csv`, and your task is to produce a single JSON report at `/app/output/eda_report.json`.

Formulas:
- `conversion_rate = conversions / users`
- `relative_difference = (treatment_conversion_rate - control_conversion_rate) / control_conversion_rate`

Expected JSON schema for `/app/output/eda_report.json`:
- `dataset_path`: string, absolute path to the analyzed CSV file
- `overall`: object containing
  - `control_users`: integer
  - `treatment_users`: integer
  - `control_conversions`: integer
  - `treatment_conversions`: integer
  - `control_conversion_rate`: number
  - `treatment_conversion_rate`: number
  - `relative_difference`: number
  - `p_value`: number
  - `ci_lower`: number
  - `ci_upper`: number
- `channels`: array of objects, one per channel, each containing
  - `channel`: string
  - `control_users`: integer
  - `treatment_users`: integer
  - `control_conversion_rate`: number
  - `treatment_conversion_rate`: number
  - `relative_difference`: number
  - `assignment_ratio`: object with
    - `control`: number
    - `treatment`: number
- `adjusted`: object containing
  - `method`: string
  - `estimate`: number
  - `ci_lower`: number
  - `ci_upper`: number
  - `p_value`: number
  - `significant`: boolean
  - `irregular_channel`: string

The report must include these fields exactly and no extra procedural details about how the data should be investigated.
