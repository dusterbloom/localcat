"""
Centralized logging configuration for LocalCat voice agent.

This module provides a unified logging configuration using loguru that ensures
all logs are properly captured, formatted, and written to the expected locations.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from loguru import logger


class LoggingConfig:
    """Centralized logging configuration manager."""

    def __init__(self):
        self.log_dir = Path("/Users/peppi/Library/Logs/LocalCat")
        self.server_log_file = self.log_dir / "server.log"
        self.configured = False

    def ensure_log_directory(self):
        """Ensure the log directory exists with proper permissions."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            # Ensure directory has proper permissions
            os.chmod(self.log_dir, 0o755)
        except Exception as e:
            print(f"Failed to create log directory {self.log_dir}: {e}")
            # Fallback to current directory
            self.log_dir = Path("./logs")
            self.server_log_file = self.log_dir / "server.log"
            self.log_dir.mkdir(exist_ok=True)

    def get_log_level(self) -> str:
        """Get log level from environment variable."""
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        valid_levels = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        return level if level in valid_levels else "INFO"

    def get_console_logging(self) -> bool:
        """Check if console logging should be enabled."""
        return os.getenv("LOG_CONSOLE", "true").lower() in ["true", "1", "yes"]

    def configure_logging(self,
                         log_level: Optional[str] = None,
                         enable_console: Optional[bool] = None,
                         log_file: Optional[Path] = None) -> bool:
        """
        Configure centralized logging for the application.

        Args:
            log_level: Log level to use (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_console: Whether to enable console logging
            log_file: Custom log file path

        Returns:
            True if configuration was successful, False otherwise
        """
        # Always reconfigure for development
        self.configured = False

        try:
            # Ensure log directory exists
            self.ensure_log_directory()

            # Get configuration values - default to DEBUG for development
            level = log_level or self.get_log_level()
            console_enabled = enable_console if enable_console is not None else self.get_console_logging()
            file_path = log_file or self.server_log_file

            # Remove ALL existing loguru handlers
            logger.remove()

            # ALWAYS use level to capture what is set in .env in LOG_LEVEL=INFO/DEBUG/TRACE/WARNING
            actual_level = level

            # Add file handler with maximum verbosity
            logger.add(
                str(file_path),
                level=actual_level,
                format=(
                    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                    "<level>{message}</level>"
                ),
                rotation="50 MB",  # Larger rotation for debug logs
                retention="14 days",  # Keep logs longer for debugging
                compression="zip",
                enqueue=False,  # Disable async logging - it's causing file handler to stop
                encoding="utf-8",
                backtrace=True,
                diagnose=True,
                catch=True  # Catch exceptions during logging
            )

            # Add console handler with maximum verbosity
            if console_enabled:
                logger.add(
                    sys.stderr,
                    level=actual_level,  # Use TRACE level for console too
                    format=(
                        "<green>{time:HH:mm:ss.SSS}</green> | "
                        "<level>{level: <8}</level> | "
                        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                        "<level>{message}</level>"
                    ),
                    backtrace=True,
                    diagnose=True,
                    catch=True
                )

            # Add a separate debug log file for maximum detail
            debug_log_path = file_path.parent / "debug.log"
            logger.add(
                str(debug_log_path),
                level=level,
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                    "{level: <8} | "
                    "{name}:{function}:{line} | "
                    "{message}"
                ),
                rotation="100 MB",
                retention="3 days",  # Keep debug logs for shorter time
                compression="zip",
                enqueue=False,  # Disable async logging - it's causing file handler to stop
                encoding="utf-8",
                backtrace=True,
                diagnose=True,
                catch=True
            )

            # Log successful configuration
            logger.info(f"VERBOSE LOGGING CONFIGURED - Level: {actual_level} (was requested: {level})")
            logger.info(f"File: {file_path}")
            logger.info(f"Debug File: {debug_log_path}")
            logger.info(f"Console: {console_enabled}")
            logger.info(f"Environment LOG_LEVEL: {os.getenv('LOG_LEVEL', 'Not set')}")

            self.configured = True
            return True

        except Exception as e:
            # Fallback to verbose basic logging if configuration fails
            print(f"FAILED TO CONFIGURE VERBOSE LOGGING: {e}")
            print("Falling back to maximum verbosity basic logging...")
            logger.remove()
            logger.add(sys.stderr, level=level, backtrace=True, diagnose=True)
            logger.add(str(self.server_log_file), level=level, backtrace=True, diagnose=True)
            return False

    def reconfigure_logging(self):
        """Reconfigure logging (useful for runtime changes)."""
        self.configured = False
        return self.configure_logging()

    def log_system_info(self):
        """Log system information for debugging."""
        import platform

        logger.info("=" * 60)
        logger.info("LOCALCAT VOICE AGENT STARTUP")
        logger.info("=" * 60)
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Working Directory: {os.getcwd()}")
        logger.info(f"Log File: {self.server_log_file}")
        logger.info(f"Log Level: {self.get_log_level()}")
        logger.info(f"Console Logging: {self.get_console_logging()}")
        logger.info("=" * 60)


# Global logging configuration instance
_logging_config = LoggingConfig()


def configure_logging(**kwargs) -> bool:
    """Configure centralized logging for the application."""
    return _logging_config.configure_logging(**kwargs)


def get_logging_config() -> LoggingConfig:
    """Get the global logging configuration instance."""
    return _logging_config


def log_system_info():
    """Log system information for debugging."""
    _logging_config.log_system_info()


def setup_logging_for_bot():
    """Setup logging specifically for the bot application."""
    success = configure_logging()
    if success:
        log_system_info()
        # Enable httpx logging for OpenAI SDK HTTP requests only when requested
        if os.getenv("LOG_HTTPX", "false").lower() in ("true", "1", "yes"):
            _setup_httpx_logging()
    return success


def _setup_httpx_logging():
    """
    Setup httpx logging to capture HTTP requests/responses from OpenAI SDK.

    The OpenAI Python SDK uses httpx for HTTP communication. This function
    intercepts httpx logs and routes them to loguru for visibility.
    """
    # Create an InterceptHandler to route standard logging to loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    # Set up httpx logger to use our handler
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.handlers = [InterceptHandler()]
    httpx_logger.setLevel(logging.DEBUG)  # Capture all httpx logs

    # Also capture httpcore logs (used internally by httpx)
    httpcore_logger = logging.getLogger("httpcore")
    httpcore_logger.handlers = [InterceptHandler()]
    httpcore_logger.setLevel(logging.DEBUG)

    # Also capture OpenAI SDK logs if available
    openai_logger = logging.getLogger("openai")
    openai_logger.handlers = [InterceptHandler()]
    openai_logger.setLevel(logging.DEBUG)

    # Enable httpx event hooks for detailed logging
    import os
    os.environ["HTTPX_LOG_LEVEL"] = "debug"

    logger.info("📡 HTTPX/HTTPCore logging enabled - OpenAI LLM HTTP requests will be logged")
