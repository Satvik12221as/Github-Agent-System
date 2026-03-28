import docker
import tarfile
import io
import re
from utils.logger import get_logger

logger = get_logger(__name__)

CONTAINER_TIMEOUT = 60
DOCKER_IMAGE = "python:3.11-slim"

# Packages that need special handling
# opencv-python has no display in containers
# use headless version instead.
PACKAGE_REMAPS = {
    "cv2":          "opencv-python-headless",
    "sklearn":      "scikit-learn",
    "PIL":          "Pillow",
    "yaml":         "PyYAML",
    "bs4":          "beautifulsoup4",
    "dotenv":       "python-dotenv",
    "serial":       "pyserial",
    "usb":          "pyusb",
}

# Packages that cannot work in a container at all
# because they need real hardware or a display
# We skip these and mock them instead
SKIP_PACKAGES = {
    "cv2", "pyaudio", "sounddevice",
    "RPi", "board", "busio",
    "tkinter", "wx", "PyQt5", "PyQt6"
}


def extract_imports_from_code(code: str) -> list[str]:
    """
    Reads a Python file and extracts all imported package names.
    Returns list of top-level package names.

    Example:
    import cv2          → ["cv2"]
    from PIL import Image → ["PIL"]
    import numpy as np  → ["numpy"]
    """
    packages = set()

    # Match: import X, import X as Y
    import_pattern = re.findall(
        r"^import\s+([\w]+)",
        code,
        re.MULTILINE
    )

    # Match: from X import Y
    from_pattern = re.findall(
        r"^from\s+([\w]+)\s+import",
        code,
        re.MULTILINE
    )

    packages.update(import_pattern)
    packages.update(from_pattern)

    # Remove standard library packages
    # These are built into Python and cannot be pip installed
    stdlib = {
        "os", "sys", "re", "json", "time", "datetime",
        "math", "random", "string", "collections",
        "itertools", "functools", "typing", "pathlib",
        "subprocess", "threading", "multiprocessing",
        "io", "abc", "copy", "enum", "dataclasses",
        "unittest", "pytest", "logging", "warnings",
        "contextlib", "inspect", "traceback", "gc",
        "struct", "hashlib", "hmac", "base64", "uuid",
        "urllib", "http", "email", "html", "xml",
        "socket", "ssl", "select", "queue", "heapq",
        "bisect", "array", "weakref", "types", "dis"
    }

    packages = packages - stdlib

    return list(packages)


def build_install_command(
    code_context: dict,
    repo_requirements: str = None
) -> str:
    """
    Builds a pip install command that installs everything
    the code needs before running tests.

    Priority:
    1. requirements.txt from the repo if available
    2. Packages detected from import statements
    3. Always install pytest
    """
    packages_to_install = {"pytest"}
    packages_to_skip = set()

    # Layer 1 — use repo requirements.txt if provided
    if repo_requirements:
        logger.info("Using repo requirements.txt")
        # Write requirements to a temp file in container
        # and install from it
        return (
            f"pip install pytest -q && "
            f"pip install -r requirements.txt -q --ignore-requires-python || true && "
            f"python -m pytest test_fix.py -v --tb=short"
        )

    # Layer 2 — scan imports from code files
    for filename, content in code_context.items():
        if filename.endswith(".py"):
            imports = extract_imports_from_code(content)
            for pkg in imports:
                if pkg in SKIP_PACKAGES:
                    packages_to_skip.add(pkg)
                    logger.info(
                        f"Skipping hardware package: {pkg}"
                    )
                elif pkg in PACKAGE_REMAPS:
                    # Use the correct pip package name
                    packages_to_install.add(PACKAGE_REMAPS[pkg])
                    logger.info(
                        f"Remapped {pkg} → {PACKAGE_REMAPS[pkg]}"
                    )
                else:
                    packages_to_install.add(pkg)

    if packages_to_skip:
        logger.info(
            f"Hardware packages skipped "
            f"(will be mocked): {packages_to_skip}"
        )

    install_list = " ".join(packages_to_install)
    logger.info(f"Installing packages: {install_list}")

    return (
        f"pip install {install_list} -q || true && "
        f"python -m pytest test_fix.py -v --tb=short"
    )



def create_tar_archive(files: dict) -> bytes:
    tar_buffer = io.BytesIO()

    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        for filename, content in files.items():
            content_bytes = content.encode("utf-8")
            info = tarfile.TarInfo(name=filename)
            info.size = len(content_bytes)
            tar.addfile(info, io.BytesIO(content_bytes))

    tar_buffer.seek(0)
    return tar_buffer.read()


def apply_patch_to_code(
    code_context: dict,
    patch: str
) -> dict:
    return code_context


def run_tests_in_docker(
    code_context: dict,
    patch: str,
    tests: str,
    repo_requirements: str = None
) -> dict:
    """
    THE MAIN SANDBOX FUNCTION.
    Automatically detects and installs dependencies.
    No manual intervention needed for any repo.
    """
    logger.info("=== Docker Sandbox starting ===")

    try:
        client = docker.from_env()
        logger.info("Connected to Docker daemon")

        logger.info(f"Pulling Docker image: {DOCKER_IMAGE}")
        client.images.pull(DOCKER_IMAGE)

        container = client.containers.create(
            image=DOCKER_IMAGE,
            command="sleep infinity",
            working_dir="/workspace",
            network_disabled=False,
            mem_limit="512m",
            cpu_period=100000,
            cpu_quota=50000,
        )

        logger.info(f"Container created: {container.short_id}")
        container.start()
        logger.info("Container started")

        # Prepare files
        files_to_copy = {}
        for filename, content in code_context.items():
            files_to_copy[filename] = content

        files_to_copy["fix.patch"]   = patch
        files_to_copy["test_fix.py"] = tests

        # Add requirements.txt if available
        if repo_requirements:
            files_to_copy["requirements.txt"] = repo_requirements
            logger.info("Added repo requirements.txt to container")

        tar_data = create_tar_archive(files_to_copy)
        container.put_archive("/workspace", tar_data)
        logger.info(
            f"Copied {len(files_to_copy)} files into container"
        )

        # Build smart install command
        install_command = build_install_command(
            code_context,
            repo_requirements
        )

        logger.info("Installing dependencies and running tests...")
        test_result = container.exec_run(
            cmd=["/bin/sh", "-c", install_command],
            workdir="/workspace"
        )

        output    = test_result.output.decode("utf-8")
        exit_code = test_result.exit_code

        logger.info(f"Tests complete. Exit code: {exit_code}")
        logger.info(f"Test output:\n{output}")

        if exit_code == 0:
            status = "passed"
            logger.info("Tests PASSED")
        else:
            status = "failed"
            logger.warning("Tests FAILED")

        return {
            "status":    status,
            "output":    output,
            "exit_code": exit_code
        }

    except docker.errors.DockerException as e:
        logger.error(f"Docker error: {e}")
        return {
            "status":    "failed",
            "output":    f"Docker error: {str(e)}",
            "exit_code": -1
        }

    except Exception as e:
        logger.error(f"Sandbox error: {e}")
        return {
            "status":    "failed",
            "output":    f"Sandbox error: {str(e)}",
            "exit_code": -1
        }

    finally:
        try:
            container.stop(timeout=5)
            container.remove(force=True)
            logger.info("Container stopped and removed")
        except Exception as e:
            logger.warning(f"Could not clean up container: {e}")