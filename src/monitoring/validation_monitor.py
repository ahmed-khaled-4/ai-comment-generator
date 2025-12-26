"""
Validation Monitoring System

This module tracks:
- Safety violations
- Rejection rates
- Retry patterns
- Generates alerts for unusual patterns
- Logs metrics and generates summary reports
"""

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SafetyViolation:
    """Record of a safety violation."""
    timestamp: str
    violation_type: str
    comment: str
    severity: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RejectionRecord:
    """Record of a rejection."""
    timestamp: str
    reason: str
    comment: str
    attempt_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryRecord:
    """Record of a retry attempt."""
    timestamp: str
    attempt_number: int
    success: bool
    delay_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidationMonitor:
    """
    Monitoring system for validation and safety tracking.
    
    This class tracks:
    - Safety violations
    - Rejection rates
    - Retry patterns
    - Generates alerts for unusual patterns
    """
    
    def __init__(self, log_dir: str = "logs/validation"):
        """
        Initialize validation monitor.
        
        Args:
            log_dir: Directory to store monitoring logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.safety_violations: List[SafetyViolation] = []
        self.rejections: List[RejectionRecord] = []
        self.retries: List[RetryRecord] = []
        
        # Alert thresholds
        self.violation_rate_threshold = 0.1  # 10% violation rate
        self.rejection_rate_threshold = 0.2  # 20% rejection rate
        self.retry_rate_threshold = 0.3  # 30% retry rate
        
        logger.info(f"ValidationMonitor initialized (log_dir: {self.log_dir})")
    
    def log_safety_violation(self,
                            violation_type: str,
                            comment: str,
                            severity: str = "medium",
                            metadata: Optional[Dict[str, Any]] = None):
        """
        Log a safety violation.
        
        Args:
            violation_type: Type of violation (e.g., "profanity", "pii", "code_injection")
            comment: Comment that violated safety rules
            severity: Severity level ("low", "medium", "high")
            metadata: Optional metadata about the violation
        """
        violation = SafetyViolation(
            timestamp=datetime.now().isoformat(),
            violation_type=violation_type,
            comment=comment[:200] + '...' if len(comment) > 200 else comment,
            severity=severity,
            metadata=metadata or {}
        )
        self.safety_violations.append(violation)
        
        # Check for alerts
        self._check_violation_alerts()
        
        logger.warning(f"Safety violation logged: {violation_type} (severity: {severity})")
    
    def log_rejection(self,
                     reason: str,
                     comment: str,
                     attempt_number: int = 1,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Log a rejection.
        
        Args:
            reason: Reason for rejection
            comment: Rejected comment
            attempt_number: Attempt number when rejected
            metadata: Optional metadata
        """
        rejection = RejectionRecord(
            timestamp=datetime.now().isoformat(),
            reason=reason,
            comment=comment[:200] + '...' if len(comment) > 200 else comment,
            attempt_number=attempt_number,
            metadata=metadata or {}
        )
        self.rejections.append(rejection)
        
        # Check for alerts
        self._check_rejection_alerts()
        
        logger.info(f"Rejection logged: {reason} (attempt: {attempt_number})")
    
    def log_retry(self,
                  attempt_number: int,
                  success: bool,
                  delay_seconds: float = 0.0,
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Log a retry attempt.
        
        Args:
            attempt_number: Attempt number
            success: Whether retry was successful
            delay_seconds: Delay before retry
            metadata: Optional metadata
        """
        retry = RetryRecord(
            timestamp=datetime.now().isoformat(),
            attempt_number=attempt_number,
            success=success,
            delay_seconds=delay_seconds,
            metadata=metadata or {}
        )
        self.retries.append(retry)
        
        # Check for alerts
        self._check_retry_alerts()
        
        logger.debug(f"Retry logged: attempt {attempt_number}, success: {success}")
    
    def get_statistics(self, time_window_hours: Optional[int] = None) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Args:
            time_window_hours: Optional time window in hours (None = all time)
        
        Returns:
            Dictionary with statistics
        """
        cutoff_time = None
        if time_window_hours:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Filter by time window
        violations = self._filter_by_time(self.safety_violations, cutoff_time)
        rejections = self._filter_by_time(self.rejections, cutoff_time)
        retries = self._filter_by_time(self.retries, cutoff_time)
        
        # Calculate statistics
        total_violations = len(violations)
        total_rejections = len(rejections)
        total_retries = len(retries)
        
        # Violation breakdown by type
        violation_types = defaultdict(int)
        for v in violations:
            violation_types[v.violation_type] += 1
        
        # Violation breakdown by severity
        violation_severities = defaultdict(int)
        for v in violations:
            violation_severities[v.severity] += 1
        
        # Rejection breakdown by reason
        rejection_reasons = defaultdict(int)
        for r in rejections:
            rejection_reasons[r.reason] += 1
        
        # Retry statistics
        successful_retries = sum(1 for r in retries if r.success)
        failed_retries = total_retries - successful_retries
        retry_success_rate = (successful_retries / total_retries * 100) if total_retries > 0 else 0
        
        # Average retry delay
        avg_retry_delay = sum(r.delay_seconds for r in retries) / total_retries if total_retries > 0 else 0
        
        return {
            'time_window_hours': time_window_hours,
            'total_violations': total_violations,
            'violation_types': dict(violation_types),
            'violation_severities': dict(violation_severities),
            'total_rejections': total_rejections,
            'rejection_reasons': dict(rejection_reasons),
            'total_retries': total_retries,
            'successful_retries': successful_retries,
            'failed_retries': failed_retries,
            'retry_success_rate': round(retry_success_rate, 2),
            'average_retry_delay_seconds': round(avg_retry_delay, 3),
        }
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a summary report.
        
        Args:
            output_path: Optional path to save report (if None, returns as string)
        
        Returns:
            Report as string
        """
        stats = self.get_statistics()
        
        report_lines = [
            "=" * 60,
            "VALIDATION MONITORING REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            "",
            "SAFETY VIOLATIONS",
            "-" * 60,
            f"Total violations: {stats['total_violations']}",
            "",
            "Violation types:",
        ]
        
        for vtype, count in stats['violation_types'].items():
            report_lines.append(f"  - {vtype}: {count}")
        
        report_lines.extend([
            "",
            "Violation severities:",
        ])
        
        for severity, count in stats['violation_severities'].items():
            report_lines.append(f"  - {severity}: {count}")
        
        report_lines.extend([
            "",
            "REJECTIONS",
            "-" * 60,
            f"Total rejections: {stats['total_rejections']}",
            "",
            "Rejection reasons:",
        ])
        
        for reason, count in stats['rejection_reasons'].items():
            report_lines.append(f"  - {reason}: {count}")
        
        report_lines.extend([
            "",
            "RETRIES",
            "-" * 60,
            f"Total retries: {stats['total_retries']}",
            f"Successful retries: {stats['successful_retries']}",
            f"Failed retries: {stats['failed_retries']}",
            f"Retry success rate: {stats['retry_success_rate']}%",
            f"Average retry delay: {stats['average_retry_delay_seconds']}s",
            "",
            "=" * 60,
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            Path(output_path).write_text(report, encoding='utf-8')
            logger.info(f"Report saved to: {output_path}")
        
        return report
    
    def _filter_by_time(self, records: List[Any], cutoff_time: Optional[datetime]) -> List[Any]:
        """Filter records by time window."""
        if not cutoff_time:
            return records
        
        filtered = []
        for record in records:
            record_time = datetime.fromisoformat(record.timestamp)
            if record_time >= cutoff_time:
                filtered.append(record)
        
        return filtered
    
    def _check_violation_alerts(self):
        """Check for violation rate alerts."""
        if len(self.safety_violations) < 10:
            return  # Not enough data
        
        # Calculate violation rate (simplified - would need total requests)
        # For now, check if recent violations exceed threshold
        recent_violations = self._filter_by_time(
            self.safety_violations,
            datetime.now() - timedelta(hours=1)
        )
        
        if len(recent_violations) > 5:  # More than 5 in last hour
            logger.warning(f"High violation rate detected: {len(recent_violations)} violations in last hour")
    
    def _check_rejection_alerts(self):
        """Check for rejection rate alerts."""
        if len(self.rejections) < 10:
            return
        
        recent_rejections = self._filter_by_time(
            self.rejections,
            datetime.now() - timedelta(hours=1)
        )
        
        if len(recent_rejections) > 10:  # More than 10 in last hour
            logger.warning(f"High rejection rate detected: {len(recent_rejections)} rejections in last hour")
    
    def _check_retry_alerts(self):
        """Check for retry rate alerts."""
        if len(self.retries) < 10:
            return
        
        recent_retries = self._filter_by_time(
            self.retries,
            datetime.now() - timedelta(hours=1)
        )
        
        failed_retries = sum(1 for r in recent_retries if not r.success)
        if len(recent_retries) > 0:
            failure_rate = failed_retries / len(recent_retries)
            if failure_rate > self.retry_rate_threshold:
                logger.warning(
                    f"High retry failure rate detected: {failure_rate:.2%} "
                    f"({failed_retries}/{len(recent_retries)})"
                )


# Global instance
_monitor_instance: Optional[ValidationMonitor] = None


def get_monitor() -> ValidationMonitor:
    """
    Get singleton instance of ValidationMonitor.
    
    Returns:
        ValidationMonitor instance
    """
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ValidationMonitor()
    return _monitor_instance


def log_safety_violation(violation_type: str,
                        comment: str,
                        severity: str = "medium",
                        metadata: Optional[Dict[str, Any]] = None):
    """Log a safety violation (convenience function)."""
    get_monitor().log_safety_violation(violation_type, comment, severity, metadata)


def log_rejection(reason: str,
                 comment: str,
                 attempt_number: int = 1,
                 metadata: Optional[Dict[str, Any]] = None):
    """Log a rejection (convenience function)."""
    get_monitor().log_rejection(reason, comment, attempt_number, metadata)


def log_retry(attempt_number: int,
             success: bool,
             delay_seconds: float = 0.0,
             metadata: Optional[Dict[str, Any]] = None):
    """Log a retry (convenience function)."""
    get_monitor().log_retry(attempt_number, success, delay_seconds, metadata)


def get_validation_statistics(time_window_hours: Optional[int] = None) -> Dict[str, Any]:
    """Get validation statistics (convenience function)."""
    return get_monitor().get_statistics(time_window_hours)


def generate_validation_report(output_path: Optional[str] = None) -> str:
    """Generate validation report (convenience function)."""
    return get_monitor().generate_report(output_path)

