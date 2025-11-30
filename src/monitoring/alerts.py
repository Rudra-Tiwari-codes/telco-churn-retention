"""
Alerting system for drift and performance degradation notifications.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


class AlertManager:
    """Manager for sending alerts via various channels."""

    def __init__(
        self,
        slack_webhook_url: str | None = None,
        email_config: dict[str, Any] | None = None,
        alert_file: Path | None = None,
    ) -> None:
        """Initialize alert manager.

        Args:
            slack_webhook_url: Slack webhook URL for notifications. If None, will try to load from SLACK_WEBHOOK_URL env var.
            email_config: Email configuration dict with keys: smtp_server, smtp_port,
                         username, password, from_email, to_emails. If None, will try to load from env vars.
            alert_file: Optional file path to log alerts (for testing/debugging).
        """
        # Load from environment variables if not provided
        self.slack_webhook_url = slack_webhook_url or os.getenv("SLACK_WEBHOOK_URL")

        # Load email config from environment if not provided
        if email_config is None:
            email_config = self._load_email_config_from_env()
        self.email_config = email_config

        self.alert_file = alert_file

    def _load_email_config_from_env(self) -> dict[str, Any] | None:
        """Load email configuration from environment variables.

        Returns:
            Email config dict or None if not all required vars are set.
        """
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = os.getenv("SMTP_PORT")
        smtp_username = os.getenv("SMTP_USERNAME")
        smtp_password = os.getenv("SMTP_PASSWORD")
        smtp_from = os.getenv("SMTP_FROM_EMAIL")
        smtp_to = os.getenv("SMTP_TO_EMAILS")

        if not all([smtp_server, smtp_port, smtp_from, smtp_to]):
            return None

        config: dict[str, Any] = {
            "smtp_server": smtp_server,
            "smtp_port": int(smtp_port),
            "from_email": smtp_from,
            "to_emails": [email.strip() for email in smtp_to.split(",")],
        }

        if smtp_username and smtp_password:
            config["username"] = smtp_username
            config["password"] = smtp_password

        return config

    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Send alert via configured channels.

        Args:
            title: Alert title.
            message: Alert message.
            severity: Alert severity ("low", "medium", "high", "critical").
            metadata: Optional metadata dictionary.

        Returns:
            True if alert was sent successfully.
        """
        success = True

        # Log alert to file if configured
        if self.alert_file:
            self._log_alert_to_file(title, message, severity, metadata)

        # Send Slack alert
        if self.slack_webhook_url:
            try:
                self._send_slack_alert(title, message, severity, metadata)
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")
                success = False

        # Send email alert
        if self.email_config:
            try:
                self._send_email_alert(title, message, severity, metadata)
            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")
                success = False

        return success

    def _send_slack_alert(
        self,
        title: str,
        message: str,
        severity: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Send alert to Slack via webhook."""
        # Determine color based on severity
        color_map = {
            "low": "#36a64f",  # Green
            "medium": "#ff9900",  # Orange
            "high": "#ff0000",  # Red
            "critical": "#8b0000",  # Dark red
        }
        color = color_map.get(severity, "#808080")  # Gray default

        # Build Slack message
        slack_message = {
            "attachments": [
                {
                    "color": color,
                    "title": title,
                    "text": message,
                    "fields": [],
                    "footer": "Telco Churn Monitoring",
                    "ts": int(Path().stat().st_mtime) if Path().exists() else None,
                }
            ]
        }

        # Add metadata as fields
        if metadata:
            for key, value in metadata.items():
                slack_message["attachments"][0]["fields"].append(
                    {
                        "title": str(key),
                        "value": str(value),
                        "short": True,
                    }
                )

        # Send to Slack
        response = requests.post(
            self.slack_webhook_url,
            json=slack_message,
            timeout=10,
        )
        response.raise_for_status()

    def _send_email_alert(
        self,
        title: str,
        message: str,
        severity: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Send alert via email."""
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        if not self.email_config:
            return

        # Build email
        msg = MIMEMultipart()
        msg["From"] = self.email_config["from_email"]
        msg["To"] = ", ".join(self.email_config["to_emails"])
        msg["Subject"] = f"[{severity.upper()}] {title}"

        # Build body
        body = f"{message}\n\n"
        if metadata:
            body += "Details:\n"
            for key, value in metadata.items():
                body += f"  {key}: {value}\n"

        msg.attach(MIMEText(body, "plain"))

        # Send email
        server = smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"])
        server.starttls()
        if "username" in self.email_config and "password" in self.email_config:
            server.login(self.email_config["username"], self.email_config["password"])
        server.send_message(msg)
        server.quit()

    def _log_alert_to_file(
        self,
        title: str,
        message: str,
        severity: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Log alert to file for debugging."""
        alert_data = {
            "title": title,
            "message": message,
            "severity": severity,
            "metadata": metadata,
        }

        self.alert_file.parent.mkdir(parents=True, exist_ok=True)

        # Append to file
        alerts = []
        if self.alert_file.exists():
            with open(self.alert_file) as f:
                alerts = json.load(f)

        alerts.append(alert_data)

        # Keep only last 1000 alerts
        alerts = alerts[-1000:]

        with open(self.alert_file, "w") as f:
            json.dump(alerts, f, indent=2)

    def alert_on_drift(
        self,
        drift_report: Any,  # DriftReport from drift.py
        threshold_severity: str = "low",
    ) -> bool:
        """Send alert if drift is detected.

        Args:
            drift_report: DriftReport object.
            threshold_severity: Minimum severity to trigger alert.

        Returns:
            True if alert was sent.
        """
        if not drift_report.overall_drift_detected:
            return False

        # Determine if we should alert based on severity
        severity_map = {"none": 0, "low": 1, "medium": 2, "high": 3}
        threshold_level = severity_map.get(threshold_severity, 1)

        # Find highest severity drift
        max_severity = "none"
        for metric in drift_report.data_drift:
            if severity_map.get(metric.drift_severity, 0) > severity_map.get(max_severity, 0):
                max_severity = metric.drift_severity

        if drift_report.prediction_drift:
            pred_severity = drift_report.prediction_drift.drift_severity
            if severity_map.get(pred_severity, 0) > severity_map.get(max_severity, 0):
                max_severity = pred_severity

        if drift_report.label_drift:
            label_severity = drift_report.label_drift.drift_severity
            if severity_map.get(label_severity, 0) > severity_map.get(max_severity, 0):
                max_severity = label_severity

        if severity_map.get(max_severity, 0) < threshold_level:
            return False

        # Build alert message
        title = "Data Drift Detected"
        message = f"Drift detected with severity: {max_severity}\n"
        if drift_report.drift_summary:
            message += f"Features with drift: {drift_report.drift_summary.get('features_with_drift', 0)}/{drift_report.drift_summary.get('total_features_checked', 0)}\n"
            if drift_report.prediction_drift and drift_report.prediction_drift.drift_detected:
                message += "Prediction drift detected\n"
            if drift_report.label_drift and drift_report.label_drift.drift_detected:
                message += "Label drift detected\n"

        metadata = {
            "timestamp": drift_report.timestamp,
            "drift_summary": drift_report.drift_summary,
        }

        alert_severity = "high" if max_severity == "high" else "medium"
        return self.send_alert(title, message, severity=alert_severity, metadata=metadata)

    def alert_on_performance_degradation(
        self,
        performance_report: Any,  # PerformanceReport from performance.py
        threshold_severity: str = "low",
    ) -> bool:
        """Send alert if performance degradation is detected.

        Args:
            performance_report: PerformanceReport object.
            threshold_severity: Minimum severity to trigger alert.

        Returns:
            True if alert was sent.
        """
        if not performance_report.performance_degradation:
            return False

        severity_map = {"none": 0, "low": 1, "medium": 2, "high": 3}
        threshold_level = severity_map.get(threshold_severity, 1)
        current_level = severity_map.get(performance_report.degradation_severity, 0)

        if current_level < threshold_level:
            return False

        # Build alert message
        title = "Model Performance Degradation Detected"
        message = f"Performance degradation detected with severity: {performance_report.degradation_severity}\n"

        if performance_report.metric_changes:
            message += "Metric changes:\n"
            for metric, change in performance_report.metric_changes.items():
                message += f"  {metric}: {change:.4f}\n"

        metadata = {
            "current_metrics": performance_report.current_metrics.to_dict(),
            "baseline_metrics": (
                performance_report.baseline_metrics.to_dict()
                if performance_report.baseline_metrics
                else None
            ),
            "metric_changes": performance_report.metric_changes,
        }

        alert_severity = (
            "high"
            if performance_report.degradation_severity == "high"
            else performance_report.degradation_severity
        )
        return self.send_alert(title, message, severity=alert_severity, metadata=metadata)
