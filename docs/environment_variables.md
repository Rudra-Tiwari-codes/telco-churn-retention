# Environment Variables

This document describes all environment variables used by the Telco Churn Retention Platform.

## Quick Start

1. Copy `.env.example` to `.env`
2. Fill in the values for your environment
3. The application will automatically load variables from `.env` if using a package like `python-dotenv`

## Model Configuration

### `MODEL_DIR`
- **Description**: Directory containing trained models
- **Default**: `models`
- **Example**: `models` or `/app/models`

### `PREDICTION_THRESHOLD`
- **Description**: Prediction threshold for binary classification (0.0 to 1.0)
- **Default**: `0.5`
- **Example**: `0.5`

## Logging

### `LOG_LEVEL`
- **Description**: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Default**: `INFO`
- **Example**: `INFO` or `DEBUG`

## MLflow Tracking

### `MLFLOW_TRACKING_URI`
- **Description**: MLflow tracking server URI
- **Default**: `file:./mlruns` (local file system)
- **Example**: 
  - Local: `file:./mlruns`
  - Remote: `http://mlflow-server:5000`
  - Databricks: `databricks://profile-name`

## API Configuration

### `API_HOST`
- **Description**: Host to bind the API server to
- **Default**: `0.0.0.0`
- **Example**: `0.0.0.0` or `127.0.0.1`

### `API_PORT`
- **Description**: Port to bind the API server to
- **Default**: `8000`
- **Example**: `8000`

### `CORS_ORIGINS`
- **Description**: Comma-separated list of allowed CORS origins, or `*` for all (development only)
- **Default**: `*`
- **Example**: 
  - Development: `*`
  - Production: `https://example.com,https://app.example.com`

## Kafka Configuration (Streaming Pipeline)

### `KAFKA_BOOTSTRAP_SERVERS`
- **Description**: Comma-separated list of Kafka broker addresses
- **Default**: `localhost:9092`
- **Example**: `localhost:9092` or `kafka1:9092,kafka2:9092`

### `KAFKA_TOPIC`
- **Description**: Kafka topic to consume customer events from
- **Default**: `customer_events`
- **Example**: `customer_events`

## Redis Configuration (Streaming Pipeline)

### `REDIS_HOST`
- **Description**: Redis server hostname
- **Default**: `localhost`
- **Example**: `localhost` or `redis-server`

### `REDIS_PORT`
- **Description**: Redis server port
- **Default**: `6379`
- **Example**: `6379`

## Alerting Configuration

### `SLACK_WEBHOOK_URL`
- **Description**: Slack webhook URL for sending alerts
- **Default**: None (alerts disabled)
- **Example**: `https://hooks.slack.com/services/YOUR/WEBHOOK/URL`

### `SMTP_SERVER`
- **Description**: SMTP server hostname for email alerts
- **Default**: None (email alerts disabled)
- **Example**: `smtp.gmail.com`

### `SMTP_PORT`
- **Description**: SMTP server port
- **Default**: None
- **Example**: `587` (TLS) or `465` (SSL)

### `SMTP_USERNAME`
- **Description**: SMTP authentication username
- **Default**: None
- **Example**: `your-email@gmail.com`

### `SMTP_PASSWORD`
- **Description**: SMTP authentication password (use app-specific password for Gmail)
- **Default**: None
- **Example**: `your-app-password`

### `SMTP_FROM_EMAIL`
- **Description**: Email address to send alerts from
- **Default**: None
- **Example**: `alerts@example.com`

### `SMTP_TO_EMAILS`
- **Description**: Comma-separated list of email addresses to send alerts to
- **Default**: None
- **Example**: `admin@example.com,team@example.com`

## Security Notes

1. **Never commit `.env` files** - They contain sensitive information
2. **Use secrets management** in production (Azure Key Vault, AWS Secrets Manager, etc.)
3. **Restrict CORS origins** in production - Never use `*` in production
4. **Use app-specific passwords** for email services like Gmail
5. **Rotate credentials regularly** - Especially for production environments

## Loading Environment Variables

The application uses standard `os.getenv()` calls. To load from `.env` files, you can use:

```python
# Option 1: Use python-dotenv (recommended)
from dotenv import load_dotenv
load_dotenv()

# Option 2: Export variables in your shell
export MODEL_DIR=/path/to/models
export PREDICTION_THRESHOLD=0.5
```

