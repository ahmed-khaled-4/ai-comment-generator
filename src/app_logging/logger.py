"""
Comprehensive logging system for AI model interactions.
Logs prompts, outputs, metadata, and failures.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import traceback
import uuid

# Configure main logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelLogger:
    """
    Comprehensive logging for AI model interactions, including inputs, outputs, metadata, and failures.
    Logs are stored in structured subdirectories and a daily complete record file.
    """
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.prompts_dir = self.log_dir / "prompts"
        self.outputs_dir = self.log_dir / "outputs"
        self.metadata_dir = self.log_dir / "metadata"
        self.failures_dir = self.log_dir / "failures"
        self.ensure_log_directories()

        self.daily_record_file = self.log_dir / f"complete_records_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.activity_log_file = self.log_dir / f"model_activity_{datetime.now().strftime('%Y%m%d')}.log"
        
        self._setup_activity_logger()
        logger.info(f"ModelLogger initialized. Logs will be stored in: {self.log_dir}")

    def ensure_log_directories(self):
        """Ensures all necessary log directories exist."""
        self.log_dir.mkdir(exist_ok=True)
        self.prompts_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        self.failures_dir.mkdir(exist_ok=True)

    def _setup_activity_logger(self):
        """Sets up a separate logger for general model activity."""
        self.activity_logger = logging.getLogger('ModelActivity')
        self.activity_logger.setLevel(logging.INFO)
        
        # Prevent adding multiple handlers if already exists
        if not self.activity_logger.handlers:
            file_handler = logging.FileHandler(self.activity_log_file)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.activity_logger.addHandler(file_handler)
            # Also log to console
            self.activity_logger.addHandler(logging.StreamHandler())

    def log_generation(
        self,
        request_id: str,
        code: str,
        language: str,
        comment_type: str,
        prompt: str,
        generated_comment: str,
        metadata: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None
    ):
        """
        Logs a complete generation request and its outcome.
        """
        timestamp = datetime.now().isoformat()
        log_entry = {
            "request_id": request_id,
            "timestamp": timestamp,
            "success": success,
            "language": language,
            "comment_type": comment_type,
            "code": code,
            "prompt": prompt,
            "generated_comment": generated_comment,
            "error": error,
            "metadata": metadata
        }

        # Log to daily complete records
        with open(self.daily_record_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Log individual components
        (self.prompts_dir / f"{request_id}.txt").write_text(prompt, encoding="utf-8")
        (self.outputs_dir / f"{request_id}.txt").write_text(generated_comment, encoding="utf-8")
        (self.metadata_dir / f"{request_id}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if not success and error:
            (self.failures_dir / f"{request_id}.json").write_text(json.dumps(log_entry, indent=2), encoding="utf-8")
            self.activity_logger.error(f"Generation {request_id}: FAILED - {language} {comment_type} - {error}")
        else:
            self.activity_logger.info(f"Generation {request_id}: SUCCESS - {language} {comment_type}")

    def log_error(
        self,
        request_id: str,
        code: str,
        language: str,
        comment_type: str,
        prompt: str,
        error_message: str,
        metadata: Dict[str, Any]
    ):
        """
        Logs an error during generation.
        """
        self.log_generation(
            request_id=request_id,
            code=code,
            language=language,
            comment_type=comment_type,
            prompt=prompt,
            generated_comment="",  # No comment generated on error
            metadata=metadata,
            success=False,
            error=error_message
        )
        self.activity_logger.error(f"Error {request_id}: {error_message}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Aggregates and returns statistics from the logs.
        """
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        total_latency = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_total_tokens = 0
        language_counts: Dict[str, int] = {}
        comment_type_counts: Dict[str, int] = {}
        
        # Read all complete records for statistics
        all_records = []
        for log_file in self.log_dir.glob("complete_records_*.jsonl"):
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed log line in {log_file}: {line.strip()}")

        for entry in all_records:
            total_requests += 1
            if entry.get("success"):
                successful_requests += 1
            else:
                failed_requests += 1

            metadata = entry.get("metadata", {})
            total_latency += metadata.get("latency", 0.0)
            total_prompt_tokens += metadata.get("prompt_tokens", 0) or 0
            total_completion_tokens += metadata.get("completion_tokens", 0) or 0
            total_total_tokens += metadata.get("total_tokens", 0) or 0

            lang = entry.get("language", "unknown")
            language_counts[lang] = language_counts.get(lang, 0) + 1

            c_type = entry.get("comment_type", "unknown")
            comment_type_counts[c_type] = comment_type_counts.get(c_type, 0) + 1

        avg_latency = total_latency / total_requests if total_requests > 0 else 0
        avg_prompt_tokens = total_prompt_tokens / total_requests if total_requests > 0 else 0
        avg_completion_tokens = total_completion_tokens / total_requests if total_requests > 0 else 0
        avg_total_tokens = total_total_tokens / total_requests if total_requests > 0 else 0
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0

        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": round(success_rate, 2),
            "average_latency_seconds": round(avg_latency, 3),
            "average_prompt_tokens": round(avg_prompt_tokens),
            "average_completion_tokens": round(avg_completion_tokens),
            "average_total_tokens": round(avg_total_tokens),
            "language_distribution": language_counts,
            "comment_type_distribution": comment_type_counts,
            "log_files": [str(f.name) for f in self.log_dir.glob("*.jsonl")]
        }

    def get_latest_logs(self, n: int = 10) -> List[Dict[str, Any]]:
        """Retrieves the latest N log entries from the complete records."""
        all_records = []
        for log_file in sorted(self.log_dir.glob("complete_records_*.jsonl")):
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return all_records[-n:]

    def get_log_by_request_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific log entry by request ID."""
        for log_file in self.log_dir.glob("complete_records_*.jsonl"):
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("request_id") == request_id:
                            return entry
                    except json.JSONDecodeError:
                        continue
        return None


# Singleton instance
_model_logger_instance: Optional[ModelLogger] = None


def get_logger() -> ModelLogger:
    """Returns a singleton instance of ModelLogger."""
    global _model_logger_instance
    if _model_logger_instance is None:
        _model_logger_instance = ModelLogger()
    return _model_logger_instance


def log_generation_wrapper(
    request_id: str,
    code: str,
    language: str,
    comment_type: str,
    prompt: str,
    generated_comment: str,
    metadata: Dict[str, Any],
    success: bool = True,
    error: Optional[str] = None
):
    """Wrapper to log generation using the singleton logger instance."""
    get_logger().log_generation(
        request_id, code, language, comment_type, prompt, generated_comment, metadata, success, error
    )


def log_error_wrapper(
    request_id: str,
    code: str,
    language: str,
    comment_type: str,
    prompt: str,
    error_message: str,
    metadata: Dict[str, Any]
):
    """Wrapper to log errors using the singleton logger instance."""
    get_logger().log_error(
        request_id, code, language, comment_type, prompt, error_message, metadata
    )
