# Dataset Documentation

## Source

**Dataset:** Sov.ai Earnings Surprise  
**Repository:** https://huggingface.co/datasets/sovai/earnings_surprise  
**Format:** Apache Parquet  
**Source revision:** `15d40bd943d249f4b94d866d92986addb53d37cc`

The dataset is used to investigate whether historical earnings patterns contain information about a company's next recorded earnings outcome.

## Reproduction

Download and verify the source dataset:

```bash
python -m earnings_surprise.data.download
```

Run the initial inspection:

```bash
python -m earnings_surprise.data.inspect
```

The inspection produces `results/data_inspection.json` which contains source identification, schema information, missing-value counts, date coverage and preliminary event-key statistics.

## Data Handling

Raw dataset files are stored under `data/raw/` and excluded from Git.

## Inspection Findings

The initial inspection successfully reproduced the raw dataset
statistics reported in the approved project proposal.

### Statistic = Verified Result** 
Raw records = 1,657,661
Columns = 7
Unique tickers = 5,739
Candidate ticker/publication pairs = 111,989
Repeated candidate-key rows = 1,511,394
Rows without usable candidate keys = 34,278

### Missing Values and Date Coverage

The initial inspection identified the following missing values:

Column = Missing Values
`ticker` = 0
`date` = 0 
`surprise_probability` = 0
`eps_surprise` = 34,291
`actual_earning_result` = 34,291
`estimated_earning` = 472
`date_pub` = 34,278 

The `date` field covers December 30 2016 through April 26 2024. The `date_pub` field covers December 24 2016 through October 30 2024.

All 34,278 records without a usable candidate key are missing publication dates.