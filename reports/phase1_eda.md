# Phase 1 EDA Snapshot

## Dataset Overview
- Rows: 7,043
- Columns: 21

## Churn Distribution
| Label | Count | Share |
| --- | --- | --- |
| No | 5174 | 0.735 |
| Yes | 1869 | 0.265 |

## Numeric Summary
| Column | Mean | Std | Min | Median | Max |
| --- | --- | --- | --- | --- | --- |
| SeniorCitizen | 0.1621468124378816 | 0.3686116056100131 | 0.0 | 0.0 | 1.0 |
| tenure | 32.37114865824223 | 24.55948102309446 | 0.0 | 29.0 | 72.0 |
| MonthlyCharges | 64.76169246059918 | 30.090047097678493 | 18.25 | 70.35 | 118.75 |
| TotalCharges | 2283.3004408418656 | 2266.771361883145 | 18.8 | 1397.475 | 8684.8 |

## Missingness
| Column | Missing Share |
| --- | --- | --- |
| TotalCharges | 0.002 |

## Categorical Cardinality (<=50 uniques)
| Column | Unique Values |
| --- | --- |
| gender | 2 |
| Partner | 2 |
| Dependents | 2 |
| PhoneService | 2 |
| MultipleLines | 3 |
| InternetService | 3 |
| OnlineSecurity | 3 |
| OnlineBackup | 3 |
| DeviceProtection | 3 |
| TechSupport | 3 |
| StreamingTV | 3 |
| StreamingMovies | 3 |
| Contract | 3 |
| PaperlessBilling | 2 |
| PaymentMethod | 4 |
| Churn | 2 |