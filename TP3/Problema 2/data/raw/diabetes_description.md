# Dataset Description: Diabetes or Prediabetes Diagnosis

## Overview
The Behavioral Risk Factor Surveillance System (BRFSS) is a health-related telephone survey conducted annually by the CDC. This dataset contains health indicators related to diabetes diagnosis and risk factors. The features are either questions directly asked of participants or calculated variables based on individual participant responses. This particular dataset includes responses from the CDC's BRFSS 2015 survey.

## Data Source
- [CDC – 2014 BRFSS Survey Data and Documentation](https://www.cdc.gov/brfss/annual_data/annual_2014.html)

## Dataset Information
- **File Names**:
    - `diabetes_dev.csv`: Development dataset file.
    - `diabetes_test.csv`: Test dataset file.
- **Purpose**: Dataset for machine learning analysis to predict diabetes status and explore associated health indicators.
- **Format**: Comma-separated values (CSV)
- **Variables**:
  - **Diabetes**: Target variable. 0 = no diabetes, 1 = prediabetes, 2 = diabetes.
  - **High_Blood_Pressure**: 0 = no high blood pressure, 1 = high blood pressure.
  - **High_Cholesterol**: 0 = no high cholesterol, 1 = high cholesterol.
  - **Cholesterol_Check**: 0 = no cholesterol check in 5 years, 1 = cholesterol check in 5 years.
  - **BMI**: Body Mass Index.
  - **Smoker**: 0 = did not smoke at least 100 cigarettes in lifetime, 1 = smoked at least 100 cigarettes.
  - **Stroke**: 0 = no stroke, 1 = had a stroke.
  - **Heart_Disease_or_Attack**: 0 = no coronary heart disease or myocardial infarction, 1 = coronary heart disease or myocardial infarction.
  - **Physical_Activity**: 0 = no physical activity in past 30 days, 1 = physical activity in past 30 days (not including job).
  - **Fruits**: 0 = does not consume fruit 1 or more times per day, 1 = consumes fruit 1 or more times per day.
  - **Veggies**: 0 = does not consume vegetables 1 or more times per day, 1 = consumes vegetables 1 or more times per day.
  - **Heavy_Alcohol_Consumption**: 0 = not a heavy drinker, 1 = heavy drinker (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week).
  - **Healthcare_Coverage**: 0 = no health care coverage, 1 = has health care coverage (including health insurance, prepaid plans, etc.).
  - **No_Dr_Visit_bc_Cost**: 0 = did not need to see a doctor due to cost, 1 = needed to see a doctor but could not due to cost.
  - **General_Health**: Self-reported general health status (scale 1-5): 1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor.
  - **Mental_Health**: Number of days in the past 30 where mental health was not good (scale 1-30).
  - **Physical_Health**: Number of days in the past 30 where physical health was not good (scale 1-30).
  - **Difficulty_Walking**: 0 = no serious difficulty walking or climbing stairs, 1 = serious difficulty walking or climbing stairs.
  - **Sex**: 0 = female, 1 = male.
  - **Age**: 13-level age category. 1 = 18-24, 9 = 60-64, 13 = 80 or older.
  - **Education**: Education level (scale 1-6): 1 = Never attended school or only kindergarten, 2 = Grades 1 through 8 (Elementary), 3 = Grades 9 through 11 (Some high school), 4 = Grade 12 or GED (High school graduate), 5 = College 1 year to 3 years (Some college or technical school), 6 = College 4 years or more (College graduate).
  - **Income**: Income level (scale 1-8): 1 = less than $10,000, 5 = less than $35,000, 8 = $75,000 or more.