# Business Intelligence Assumptions

This document describes the key assumptions and parameters used in the business intelligence modules (ROI calculator, KPI tracker, retention playbooks) for the Telco Churn Retention Platform.

## ROI Calculator Assumptions

The `ROICalculator` class (`src/business/roi.py`) uses several configurable parameters with default values:

### Customer Lifetime

- **Parameter**: `customer_lifetime_months` (default: 24 months)
- **Usage**: Used to estimate total revenue saved when preventing churn
- **Calculation**: `revenue_saved = churn_prevented × avg_monthly_revenue × customer_lifetime_months`
- **Rationale**: Assumes an average customer lifetime of 2 years. This is a conservative estimate for telco customers.
- **Customization**: Can be adjusted based on historical customer lifetime data:
  ```python
  roi_calculator = ROICalculator(customer_lifetime_months=36)  # 3 years
  ```

### Intervention Effectiveness

- **Parameter**: `intervention_effectiveness` (default: 0.30, i.e., 30%)
- **Usage**: Percentage of retention interventions that successfully prevent churn
- **Calculation**: `churn_prevented = estimated_churn × intervention_effectiveness`
- **Rationale**: Based on industry benchmarks for retention campaigns. Actual effectiveness varies by:
  - Type of intervention (discount vs. service enhancement vs. contract upgrade)
  - Customer segment (VIP vs. new customers)
  - Timing of intervention
- **Customization**: Can be adjusted based on A/B test results or historical data:
  ```python
  roi_calculator = ROICalculator(intervention_effectiveness=0.40)  # 40% success rate
  ```

### Base Churn Rate

- **Parameter**: `base_churn_rate` (default: 0.265, i.e., 26.5%)
- **Usage**: Historical baseline churn rate for comparison
- **Rationale**: Matches the actual churn rate in the Telco dataset (1869 churned / 7043 total)
- **Customization**: Should be updated based on actual business metrics:
  ```python
  roi_calculator = ROICalculator(base_churn_rate=0.20)  # 20% historical churn
  ```

## KPI Tracker Assumptions

The `KPITracker` class (`src/business/kpis.py`) uses:

### Churn Estimation Factor

- **Parameter**: `churn_estimation_factor` (default: 0.80, i.e., 80%)
- **Usage**: Multiplier to convert average churn probability into estimated actual churn volume
- **Calculation**: `estimated_monthly_churn = total_customers × avg_churn_probability × churn_estimation_factor`
- **Rationale**: Not all customers with high churn probability will actually churn. This factor accounts for:
  - Model calibration (probabilities may be overconfident)
  - External factors not captured by the model
  - Natural variation in churn behavior
- **Customization**: Should be calibrated based on historical model performance:
  ```python
  kpi_tracker = KPITracker(churn_estimation_factor=0.75)  # More conservative
  ```

### Risk Thresholds

- **Parameters**: 
  - `critical_threshold` (default: 0.75)
  - `high_threshold` (default: 0.50)
  - `medium_threshold` (default: 0.30)
- **Usage**: Churn probability thresholds for risk segmentation
- **Rationale**: Standard risk categorization for retention prioritization
- **Customization**: Can be adjusted based on business priorities:
  ```python
  kpi_tracker = KPITracker(
      critical_threshold=0.80,  # More conservative critical threshold
      high_threshold=0.60,
      medium_threshold=0.35
  )
  ```

## Retention Playbook Assumptions

The `RetentionPlaybookGenerator` class (`src/business/playbooks.py`) uses:

### Cost Parameters

- **`discount_cost_percentage`** (default: 0.15, i.e., 15% of monthly revenue)
  - Cost of discount as percentage of monthly revenue
  - Used to estimate total cost of discount-based interventions

- **`retention_specialist_cost`** (default: $50.00 per call)
  - Cost per retention specialist call/intervention
  - Includes labor and overhead

- **`account_manager_cost`** (default: $25.00 per outreach)
  - Cost per account manager proactive outreach
  - Lower than specialist cost due to less intensive engagement

### Effectiveness Estimates

The playbook generator uses fixed effectiveness estimates for different intervention types:

- **Retention specialist call**: 35% effectiveness (for critical-risk customers)
- **Personalized discount (20%)**: 45% effectiveness
- **Proactive outreach**: 25% effectiveness
- **Targeted discount (15%)**: 30% effectiveness
- **Contract upgrade incentive**: 40% effectiveness
- **Auto-pay enrollment**: 20% effectiveness
- **Service enhancement**: 15% effectiveness

**Note**: These are default estimates. In production, these should be:
- Calibrated based on historical A/B test results
- Updated regularly as intervention strategies evolve
- Segmented by customer cohort (VIP vs. new customers may respond differently)

### Lifetime Value Calculation

- **Formula**: `lifetime_value = monthly_revenue × expected_tenure_months`
- **Expected tenure**: `max(12, current_tenure + (1 - churn_probability) × 24)`
- **Rationale**: Projects future tenure based on current tenure and churn risk
- **Limitation**: Simplified projection; does not account for revenue growth or service changes

## Revenue Loss Estimation

Both ROI and KPI modules estimate revenue loss using:

```
revenue_lost = churned_customers × avg_monthly_revenue × customer_lifetime_months
```

This assumes:
- Lost revenue equals the customer's monthly charge multiplied by remaining lifetime
- No account for:
  - Customer acquisition costs (CAC)
  - Revenue growth from upsells
  - Cost savings from churned customers (if they were unprofitable)

## Recommendations for Production Use

1. **Calibrate parameters** based on historical data:
   - Measure actual intervention effectiveness via A/B tests
   - Calculate actual customer lifetime from historical cohorts
   - Track model calibration to adjust churn estimation factor

2. **Segment assumptions** by customer cohort:
   - VIP customers may have different lifetime and intervention effectiveness
   - New customers may require different thresholds and strategies

3. **Update regularly**:
   - Review and update assumptions quarterly
   - Incorporate results from retention experiments
   - Adjust thresholds based on business priorities

4. **Document customizations**:
   - When adjusting parameters, document the rationale
   - Track changes over time
   - Share assumptions with business stakeholders

## Example: Customizing for Your Business

```python
from src.business.roi import ROICalculator
from src.business.kpis import KPITracker

# Customize based on your business metrics
roi_calculator = ROICalculator(
    base_churn_rate=0.22,  # Your historical churn rate
    intervention_effectiveness=0.35,  # From A/B test results
    customer_lifetime_months=30,  # Your average customer lifetime
)

kpi_tracker = KPITracker(
    critical_threshold=0.70,  # Your risk thresholds
    high_threshold=0.50,
    medium_threshold=0.30,
    churn_estimation_factor=0.75,  # Calibrated from model performance
)
```

## References

- Industry benchmarks for retention effectiveness: Typically 20-40% depending on intervention type
- Customer lifetime in telco: Typically 18-36 months, varies by market and service type
- Model calibration: Churn estimation factors should be validated against actual churn rates over time

