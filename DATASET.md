# Dataset and Data Preparation Documentation

## Source
- **Name:** CDC Diabetes Health Indicators
- **Repository:** UCI Machine Learning Repository, dataset 891
- **Link:** https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- **Origin:** 2015 Behavioral Risk Factor Surveillance System (BRFSS), a health survey run by the US Centers for Disease Control and Prevention
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0). Confirm on the dataset page.
- **File used:** `diabetes_binary_health_indicators_BRFSS2015.csv`

## Number of original records
253,680 survey responses (real survey data, not synthetic or augmented). The notebook prints the exact count and checks that it is more than 5,000. After removing exact duplicate rows, the notebook prints the number of remaining records, which is also used for training.

## Target variable
`Diabetes_binary`: 0 = no diabetes, 1 = prediabetes or diabetes. About 14% of the records are 1, so the classes are imbalanced.

## Features used by the model (16)
| Feature | Meaning |
|---|---|
| HighBP | High blood pressure (0/1) |
| HighChol | High cholesterol (0/1) |
| BMI | Body mass index |
| Smoker | Smoked at least 100 cigarettes in life (0/1) |
| Stroke | Ever had a stroke (0/1) |
| HeartDiseaseorAttack | Coronary heart disease or heart attack (0/1) |
| PhysActivity | Physical activity in past 30 days (0/1) |
| Fruits | Eats fruit daily (0/1) |
| Veggies | Eats vegetables daily (0/1) |
| HvyAlcoholConsump | Heavy alcohol drinking (0/1) |
| GenHlth | General health, 1 (excellent) to 5 (poor) |
| MentHlth | Days of poor mental health in past 30 days |
| PhysHlth | Days of poor physical health in past 30 days |
| DiffWalk | Difficulty walking or climbing stairs (0/1) |
| Sex | 0 = female, 1 = male |
| Age | 13 age groups (1 = 18-24 years, ..., 13 = 80 or older) |

Columns not used: CholCheck, AnyHealthcare, NoDocbcCost, Education, Income. They describe healthcare access or social background, are not easy for a user to answer in a quick self-check, and could make the app unfair to some users.

## Data preparation steps
1. Load the CSV and check the number of records and the target column.
2. Inspect data types, missing values, duplicates, summary statistics and class balance.
3. Remove exact duplicate rows.
4. Confirm there are no missing values (the file has none) and that values are in valid ranges.
5. Select the 16 features and the target. All values are already numeric, so no encoding is needed. Scaling is applied inside the Logistic Regression pipeline only.
6. Split the data 80% train and 20% test with `random_state=42`, stratified on the target.
7. Train the baseline, Logistic Regression and Random Forest on the training set only. Class weights are used to handle the imbalance.
8. Evaluate every model on the test set and save the best model.

## Input handling in the application
- Age in years is converted to the dataset's 13 age groups.
- BMI is calculated from the height and weight that the user enters.
- Inputs are validated before prediction (ranges, missing values, unrealistic BMI).
