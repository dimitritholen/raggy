"""Dependency management and auto-installation."""

import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from ..config.cache import load_deps_cache, save_deps_cache
from .environment import check_environment_setup, check_uv_available


class PackageInstaller:
    """Handles package installation with caching and validation."""

    # Special cases where package name differs from import name
    IMPORT_NAME_MAP = {
        "python-magic-bin": "magic",
        "python-magic": "magic",
        "python-docx": "docx",
        "pyyaml": "yaml",
        "PyPDF2": "PyPDF2",
    }

    def __init__(self, skip_cache: bool = False) -> None:
        """Initialize installer with cache configuration.

        Args:
            skip_cache: If True, skip cache and always check/install

        """
        self.skip_cache = skip_cache
        self.cache: Dict[str, Any] = {} if skip_cache else load_deps_cache()
        self.cache_updated = False

    def install_packages(self, packages: List[str], silent_fail: bool = False) -> None:
        """Install all packages if missing.

        Args:
            packages: List of package specifications (e.g., "chromadb>=0.4.0")
            silent_fail: If True, don't print error messages on failure

        """
        self._validate_environment()

        for package_spec in packages:
            self._install_package(package_spec, silent_fail=silent_fail)

        if self.cache_updated:
            save_deps_cache(self.cache)

    def _validate_environment(self) -> None:
        """Validate UV and environment setup.

        Exits with error if validation fails.
        """
        if not check_uv_available():
            sys.exit(1)

        env_ok, env_issue = check_environment_setup()
        if not env_ok:
            self._report_env_issue(env_issue)
            sys.exit(1)

    def _report_env_issue(self, env_issue: str) -> None:
        """Report specific environment issue to user.

        Args:
            env_issue: Type of environment issue

        """
        error_messages = {
            "virtual_environment": (
                "ERROR: No virtual environment found.\n"
                "Run 'python raggy.py init' to set up the project environment."
            ),
            "pyproject": (
                "ERROR: No pyproject.toml found.\n"
                "Run 'python raggy.py init' to set up the project environment."
            ),
            "invalid_venv": (
                "ERROR: Invalid virtual environment found.\n"
                "Delete .venv directory and run 'python raggy.py init' to recreate it."
            ),
            "missing_dependencies": (
                "ERROR: Required dependencies are not installed.\n"
                "If you installed raggy as a package, run: pip install 'raggy[all]'\n"
                "If using from source, run: pip install -e '.[all]'\n"
                "Or manually install: pip install chromadb sentence-transformers PyPDF2 python-docx"
            ),
        }
        message = error_messages.get(
            env_issue, f"ERROR: Environment issue: {env_issue}"
        )
        print(message)

    def _install_package(self, package_spec: str, silent_fail: bool = False) -> None:
        """Install single package if not cached or installed.

        Args:
            package_spec: Package specification (e.g., "chromadb>=0.4.0")
            silent_fail: If True, don't print error messages on failure

        """
        package_name = self._extract_package_name(package_spec)

        # Check cache first
        if not self.skip_cache and package_name in self.cache.get("installed", {}):
            return

        # Check if already installed
        if self._is_already_installed(package_name):
            self._update_cache(package_name)
            return

        # Install the package
        self._perform_install(package_spec, package_name, silent_fail=silent_fail)

    def _extract_package_name(self, package_spec: str) -> str:
        """Extract package name from specification.

        Args:
            package_spec: Package specification like 'package>=1.0' or 'package[extra]'

        Returns:
            str: Clean package name

        """
        return package_spec.split(">=")[0].split("==")[0].split("[")[0]

    def _get_import_name(self, package_name: str) -> str:
        """Get import name for package (may differ from package name).

        Args:
            package_name: Package name as used in pip

        Returns:
            str: Import name for use with importlib

        """
        return self.IMPORT_NAME_MAP.get(package_name, package_name.replace("-", "_"))

    def _is_already_installed(self, package_name: str) -> bool:
        """Check if package is already installed.

        Args:
            package_name: Package name to check

        Returns:
            bool: True if package can be imported

        """
        import_name = self._get_import_name(package_name)
        try:
            spec = importlib.util.find_spec(import_name)
            return spec is not None
        except (ImportError, ModuleNotFoundError):
            return False

    def _update_cache(self, package_name: str) -> None:
        """Update cache with installed package timestamp.

        Args:
            package_name: Package name to cache

        """
        if "installed" not in self.cache:
            self.cache["installed"] = {}
        self.cache["installed"][package_name] = time.time()
        self.cache_updated = True

    def _perform_install(self, package_spec: str, package_name: str, silent_fail: bool = False) -> None:
        """Perform actual package installation.

        Args:
            package_spec: Full package specification for pip
            package_name: Package name for error handling
            silent_fail: If True, don't print error messages on failure

        """
        if not silent_fail:
            print(f"Installing {package_name}...")

        # Check if we're in a virtual environment
        in_venv = sys.prefix != sys.base_prefix

        try:
            if in_venv:
                # In a venv, use uv without --system flag
                subprocess.check_call(["uv", "pip", "install", package_spec],
                                    stdout=subprocess.DEVNULL if silent_fail else None,
                                    stderr=subprocess.DEVNULL if silent_fail else None)
            else:
                # Not in a venv, try uv with --system flag
                try:
                    subprocess.check_call(["uv", "pip", "install", "--system", package_spec],
                                        stdout=subprocess.DEVNULL if silent_fail else None,
                                        stderr=subprocess.DEVNULL if silent_fail else None)
                except subprocess.CalledProcessError:
                    # If uv fails, fall back to regular pip
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec],
                                        stdout=subprocess.DEVNULL if silent_fail else None,
                                        stderr=subprocess.DEVNULL if silent_fail else None)

            self._update_cache(package_name)
        except subprocess.CalledProcessError as e:
            if not silent_fail:
                print(f"Failed to install {package_name}: {e}")
            self._try_fallback_install(package_name, silent_fail=silent_fail)

    def _try_fallback_install(self, package_name: str, silent_fail: bool = False) -> None:
        """Try fallback installation for special packages.

        Args:
            package_name: Package that failed to install
            silent_fail: If True, don't print error messages on failure

        """
        if package_name != "python-magic-bin":
            return

        if not silent_fail:
            print("Trying alternative magic package...")

        # Check if we're in a virtual environment
        in_venv = sys.prefix != sys.base_prefix

        try:
            if in_venv:
                subprocess.check_call(["uv", "pip", "install", "python-magic"],
                                    stdout=subprocess.DEVNULL if silent_fail else None,
                                    stderr=subprocess.DEVNULL if silent_fail else None)
            else:
                try:
                    subprocess.check_call(["uv", "pip", "install", "--system", "python-magic"],
                                        stdout=subprocess.DEVNULL if silent_fail else None,
                                        stderr=subprocess.DEVNULL if silent_fail else None)
                except subprocess.CalledProcessError:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-magic"],
                                        stdout=subprocess.DEVNULL if silent_fail else None,
                                        stderr=subprocess.DEVNULL if silent_fail else None)

            self._update_cache(package_name)
        except subprocess.CalledProcessError:
            if not silent_fail:
                print(
                    "Warning: Could not install python-magic. "
                    "File type detection may be limited."
                )


def install_if_missing(packages: List[str], skip_cache: bool = False, silent_fail: bool = False) -> None:
    """Auto-install required packages if missing using uv.

    Args:
        packages: List of package specifications (e.g., "chromadb>=0.4.0")
        skip_cache: If True, skip cache and always check/install
        silent_fail: If True, don't print error messages on failure

    """
    installer = PackageInstaller(skip_cache=skip_cache)
    installer.install_packages(packages, silent_fail=silent_fail)


def setup_dependencies(skip_cache: bool = False, quiet: bool = False) -> None:
    """Setup dependencies with optional caching.

    Args:
        skip_cache: If True, skip cache and always check/install
        quiet: If True, suppress output (unused but kept for compatibility)

    """

    # Check if we're in a virtual environment
    env_ok, env_issue = check_environment_setup()

    if not env_ok:
        if env_issue == "missing_dependencies":
            print("\nERROR: Required dependencies are not installed.")
            print("\nIf you installed raggy as a package:")
            print("  pip install 'raggy[all]'")
            print("\nIf you're developing raggy:")
            print("  pip install -e '.[all]'")
            print("\nOr install manually:")
            print("  pip install chromadb sentence-transformers PyPDF2 python-docx")
        elif env_issue == "virtual_environment":
            print("\nERROR: Local .venv exists but is not activated.")
            print("\nPlease activate your virtual environment:")
            if sys.platform == "win32":
                print("  .venv\\Scripts\\activate")
            else:
                print("  source .venv/bin/activate")
            print("\nThen run the command again.")
        else:
            print("\nERROR: Environment is not properly set up.")
            print(f"Issue: {env_issue}")
            print("\nFor local development:")
            print("  python -m venv .venv")
            if sys.platform == "win32":
                print("  .venv\\Scripts\\activate")
            else:
                print("  source .venv/bin/activate")
            print("  pip install -e '.[all]'")
        sys.exit(1)

    # Environment is OK, proceed with dependency checks

    # Auto-install required packages if missing
    required_packages = [
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "PyPDF2>=3.0.0",
        "python-docx>=1.0.0",
    ]

    # Add optional packages (non-blocking)
    optional_packages = ["pyyaml>=6.0", "torch>=2.0.0"]

    # Platform-specific magic library is optional (for file type detection)
    if sys.platform == "win32":
        optional_packages.append("python-magic-bin>=0.4.14")
    else:
        optional_packages.append("python-magic")

    # Install required packages
    install_if_missing(required_packages, skip_cache=skip_cache)

    # Try to install optional packages but don't fail if they can't be installed
    for package in optional_packages:
        try:
            # Check if already installed before trying to install
            package_name = package.split(">=")[0].split("==")[0].split("[")[0]
            installer = PackageInstaller(skip_cache=skip_cache)
            if not installer._is_already_installed(package_name):
                # Use silent_fail=True for optional packages
                install_if_missing([package], skip_cache=skip_cache, silent_fail=True)
        except (subprocess.CalledProcessError, OSError, RuntimeError):
            # Installation failed for optional package - silently continue
            # This is expected for packages that may not be available in all environments
            pass
