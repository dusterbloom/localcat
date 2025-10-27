"""
Model pre-validation system for Kokoro TTS services.
Ensures all required model files are available and cached BEFORE Metal lock acquisition.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import hashlib
import json
from loguru import logger


class ModelValidationError(Exception):
    """Raised when model validation fails."""
    pass


class KokoroModelValidator:
    """
    Pre-validates Kokoro TTS models to ensure they're fully cached and accessible
    before attempting to initialize KPipeline inside Metal lock.
    """

    # Expected model files for different Kokoro variants
    EXPECTED_FILES = {
        "hexgrad/Kokoro-82M": [
            "config.json",
            "kokoro-v1_0.pth",  # Main model file - CRITICAL
        ],
        "mlx-community/Kokoro-82M-bf16": [
            "config.json",
            "model.safetensors",  # MLX format - CRITICAL
        ]
    }

    def __init__(self, repo_id: str = "hexgrad/Kokoro-82M"):
        self.repo_id = repo_id
        self.cache_dir = self._resolve_cache_dir()

    def _resolve_cache_dir(self) -> Optional[Path]:
        """
        Resolve the HuggingFace cache directory for this repo.

        Priority order:
        1. TAURI_RESOURCE_DIR (bundled macOS app)
        2. HUGGINGFACE_HUB_CACHE environment variable
        3. Server-relative models directory
        """
        # PRIORITY 1: Check if running in Tauri bundle
        tauri_resource_dir = os.environ.get("TAURI_RESOURCE_DIR")
        if tauri_resource_dir:
            # Bundle path: Resources/_up_/_up_/server/models/hf_cache/hub
            bundled_cache = Path(tauri_resource_dir) / "_up_" / "_up_" / "server" / "models" / "hf_cache" / "hub"
            repo_dir = bundled_cache / f"models--{self.repo_id.replace('/', '--')}"

            if repo_dir.exists():
                logger.debug(f"Using bundled cache: {repo_dir}")
                found_dir = self._find_snapshot_or_direct(repo_dir)
                if found_dir:
                    return found_dir
                logger.warning(f"Bundled cache exists but no valid model found: {repo_dir}")

        # PRIORITY 2: Environment variable
        cache_root = os.environ.get("HUGGINGFACE_HUB_CACHE")
        if cache_root:
            base = Path(cache_root)
            logger.debug(f"Using HUGGINGFACE_HUB_CACHE: {base}")
        else:
            # PRIORITY 3: Server-relative models directory
            server_root = Path(__file__).resolve().parents[2]  # Go up from core/utils/
            base = server_root / "models" / "hf_cache" / "hub"
            logger.debug(f"Using server-relative cache: {base}")

        # Construct repo directory path
        repo_dir = base / f"models--{self.repo_id.replace('/', '--')}"
        return self._find_snapshot_or_direct(repo_dir)

    def _find_snapshot_or_direct(self, repo_dir: Path) -> Optional[Path]:
        """
        Find valid model files in repo directory.
        Handles both direct model files and HuggingFace snapshot structure.
        """
        if not repo_dir.exists():
            logger.debug(f"Repo directory does not exist: {repo_dir}")
            return None

        # First, check if this is a direct model directory (has config.json)
        config_file = repo_dir / "config.json"
        if config_file.exists():
            logger.debug(f"Found direct model directory: {repo_dir}")
            return repo_dir

        # If not, check if this is a HF cache structure with snapshots
        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.exists():
            logger.debug(f"Looking for snapshots in: {snapshots_dir}")
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    # Check if this snapshot has required files
                    config_file = snapshot / "config.json"
                    if config_file.exists():
                        logger.debug(f"Found valid snapshot: {snapshot}")
                        return snapshot

            # As a last resort, check if any snapshot exists at all
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                # Return the most recent snapshot (by modification time)
                latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
                logger.debug(f"Using latest snapshot as fallback: {latest_snapshot}")
                return latest_snapshot

        logger.debug(f"No cache directory found for {self.repo_id}")
        return None

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate hash for {file_path}: {e}")
            return ""

    def validate_model_files(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that all required model files exist and are readable.

        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "missing_files": [],
            "unreadable_files": [],
            "found_files": [],
            "file_hashes": {},
            "repo_id": self.repo_id
        }

        if not self.cache_dir:
            validation_info["missing_files"] = ["cache_directory"]
            return False, validation_info

        # Get expected files for this repo
        expected_files = self.EXPECTED_FILES.get(self.repo_id, ["config.json"])

        for file_name in expected_files:
            file_path = self.cache_dir / file_name

            if not file_path.exists():
                validation_info["missing_files"].append(file_name)
                continue

            if not file_path.is_file():
                validation_info["unreadable_files"].append(file_name)
                continue

            # File exists and is readable
            validation_info["found_files"].append(file_name)

            # Calculate hash for integrity checking (optional but helpful)
            file_hash = self._calculate_file_hash(file_path)
            if file_hash:
                validation_info["file_hashes"][file_name] = file_hash

        # Model is valid if no missing or unreadable files
        is_valid = not validation_info["missing_files"] and not validation_info["unreadable_files"]

        return is_valid, validation_info

    def validate_config_json(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate config.json contains required fields for Kokoro.
        """
        config_path = self.cache_dir / "config.json" if self.cache_dir else None

        if not config_path or not config_path.exists():
            return False, {"error": "config.json not found"}

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Check for required Kokoro-specific fields based on model type
            if "hexgrad" in self.repo_id:
                # Kokoro PyTorch model has different structure
                required_fields = [
                    "dim_in",  # Input dimension
                ]

                # Optional but commonly expected fields for Kokoro
                optional_fields = [
                    "istftnet",  # ISTFT network configuration
                    "hidden_dim",
                    "dropout",
                    "max_conv_dim"
                ]
            else:
                # MLX or other models might have standard transformer structure
                required_fields = [
                    "model_type",
                    "vocab_size",
                ]

                optional_fields = [
                    "hidden_size",
                    "num_attention_heads",
                    "num_hidden_layers"
                ]

            missing_required = []
            for field in required_fields:
                if field not in config:
                    missing_required.append(field)

            config_info = {
                "has_required_fields": not missing_required,
                "missing_required_fields": missing_required,
                "model_type": config.get("model_type", config.get("dim_in", "unknown")),
                "vocab_size": config.get("vocab_size", config.get("dim_in", 0)),
                "has_optional_fields": any(field in config for field in optional_fields),
                "config_keys": list(config.keys())[:10]  # First 10 keys for debugging
            }

            return not missing_required, config_info

        except json.JSONDecodeError as e:
            return False, {"error": f"Invalid JSON in config.json: {e}"}
        except Exception as e:
            return False, {"error": f"Failed to read config.json: {e}"}

    def simulate_offline_initialization(self) -> Tuple[bool, str]:
        """
        Simulate KPipeline initialization with HF_HUB_OFFLINE=1 to catch errors
        before actual Metal lock acquisition.
        """
        if not self.cache_dir:
            return False, "Model cache directory not found"

        # Save original environment
        original_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        original_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")

        try:
            # Force offline mode
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            # Try to import and initialize KPipeline in offline mode
            from kokoro import KPipeline

            # This should work offline if files are properly cached
            logger.debug(f"Testing KPipeline initialization in offline mode...")
            test_pipeline = KPipeline(lang_code='a', repo_id=self.repo_id)

            # Test basic functionality (optional) - just verify pipeline is functional
            test_text = "Test"
            try:
                # Try a quick generation test (but don't wait for completion)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(list, test_pipeline(test_text, voice="af_heart", speed=1.0))
                    # Cancel after 1 second - we just want to test initialization
                    try:
                        future.result(timeout=1.0)
                    except concurrent.futures.TimeoutError:
                        logger.debug("KPipeline initialization test timed out (expected)")
                        future.cancel()
                logger.debug("KPipeline offline initialization test completed")
            except Exception as e:
                logger.debug(f"KPipeline generation test failed: {e}")
                # This is OK - we just wanted to test initialization

            # Clean up test pipeline
            del test_pipeline

            return True, "KPipeline initialization successful in offline mode"

        except ImportError as e:
            return False, f"Kokoro package not available: {e}"
        except Exception as e:
            error_msg = str(e)
            if "offline" in error_msg.lower() or "cache" in error_msg.lower():
                return False, f"Offline initialization failed: {error_msg}"
            return False, f"KPipeline initialization failed: {error_msg}"
        finally:
            # Restore original environment
            if original_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = original_hf_offline
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)

            if original_transformers_offline is not None:
                os.environ["TRANSFORMERS_OFFLINE"] = original_transformers_offline
            else:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)

    def comprehensive_validation(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform comprehensive model validation including file checks, config validation,
        and offline initialization simulation.
        """
        validation_result = {
            "repo_id": self.repo_id,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "file_validation": {},
            "config_validation": {},
            "offline_test": {},
            "is_valid": False,
            "errors": [],
            "warnings": []
        }

        try:
            # 1. Validate model files exist and are readable
            files_valid, files_info = self.validate_model_files()
            validation_result["file_validation"] = files_info

            if not files_valid:
                validation_result["errors"].extend(
                    f"Missing files: {files_info['missing_files']}"
                )
                validation_result["errors"].extend(
                    f"Unreadable files: {files_info['unreadable_files']}"
                )

            # 2. Validate config.json structure
            if self.cache_dir:
                config_valid, config_info = self.validate_config_json()
                validation_result["config_validation"] = config_info

                if not config_valid:
                    validation_result["errors"].append(
                        f"Config validation failed: {config_info.get('error', 'Unknown error')}"
                    )
            else:
                config_valid = False
                validation_result["errors"].append("No cache directory for config validation")

            # 3. Only test offline initialization if files are present
            if files_valid and config_valid:
                offline_valid, offline_msg = self.simulate_offline_initialization()
                validation_result["offline_test"] = {
                    "success": offline_valid,
                    "message": offline_msg
                }

                if not offline_valid:
                    validation_result["errors"].append(f"Offline initialization failed: {offline_msg}")
            else:
                validation_result["offline_test"] = {
                    "success": False,
                    "message": "Skipped due to file/config validation failures"
                }

            # Determine overall validity
            validation_result["is_valid"] = (
                files_valid and
                config_valid and
                validation_result["offline_test"]["success"]
            )

            # Add helpful information
            if validation_result["is_valid"]:
                logger.info(f"✅ Model validation successful for {self.repo_id}")
                logger.info(f"   Cache directory: {validation_result['cache_dir']}")
                logger.info(f"   Found files: {len(files_info['found_files'])}")
            else:
                logger.error(f"❌ Model validation failed for {self.repo_id}")
                for error in validation_result["errors"]:
                    logger.error(f"   - {error}")

            return validation_result["is_valid"], validation_result

        except Exception as e:
            validation_result["errors"].append(f"Validation process failed: {e}")
            logger.error(f"Model validation process failed: {e}")
            return False, validation_result


def validate_kokoro_model(repo_id: str = "hexgrad/Kokoro-82M") -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to validate a Kokoro model.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        Tuple of (is_valid, validation_result)
    """
    validator = KokoroModelValidator(repo_id)
    return validator.comprehensive_validation()


def ensure_offline_ready(repo_id: str = "hexgrad/Kokoro-82M") -> bool:
    """
    Ensure the model is ready for offline initialization.
    This is the main function to call before Metal lock acquisition.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        True if model is ready for offline initialization
    """
    is_valid, result = validate_kokoro_model(repo_id)

    if not is_valid:
        logger.error("❌ Model not ready for offline initialization:")
        for error in result["errors"]:
            logger.error(f"   - {error}")

        logger.info("💡 To fix this issue:")
        logger.info("   1. Ensure the model is downloaded and cached")
        logger.info("   2. Check that all required files are present")
        logger.info("   3. Verify cache directory permissions")
        logger.info(f"   4. Try running: python -c \"from kokoro import KPipeline; KPipeline(lang_code='a', repo_id='{repo_id}')\"")

        return False

    logger.info("✅ Model is ready for offline initialization")
    return True


if __name__ == "__main__":
    # Test validation when run directly
    import sys

    repo_id = sys.argv[1] if len(sys.argv) > 1 else "hexgrad/Kokoro-82M"

    print(f"Validating Kokoro model: {repo_id}")
    print("=" * 50)

    is_valid, result = validate_kokoro_model(repo_id)

    print(f"Validation Result: {'✅ PASS' if is_valid else '❌ FAIL'}")
    print(f"Cache Directory: {result['cache_dir']}")
    print(f"Files Found: {len(result['file_validation'].get('found_files', []))}")
    print(f"Missing Files: {result['file_validation'].get('missing_files', [])}")
    print(f"Unreadable Files: {result['file_validation'].get('unreadable_files', [])}")
    print(f"Config Valid: {result['config_validation'].get('has_required_fields', False)}")
    print(f"Offline Test: {'✅' if result['offline_test'].get('success') else '❌'}")

    if not is_valid:
        print("\nErrors:")
        for error in result["errors"]:
            print(f"  - {error}")

    sys.exit(0 if is_valid else 1)