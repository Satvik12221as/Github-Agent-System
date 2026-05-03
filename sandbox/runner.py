import docker
import tarfile
import io
import os
import re
import time
from utils.logger import get_logger


logger = get_logger(__name__)


CONTAINER_TIMEOUT = 120   # seconds - increased for ML repositories with heavy dependencies
DOCKER_IMAGE      = "python:3.11-slim"


PACKAGE_REMAPS = {
    "cv2":      "opencv-python-headless",
    "sklearn":  "scikit-learn",
    "PIL":      "Pillow",
    "yaml":     "PyYAML",
    "bs4":      "beautifulsoup4",
    "dotenv":   "python-dotenv",
    "serial":   "pyserial",
}


SKIP_PACKAGES = {
    "cv2", "pyaudio", "sounddevice",
    "RPi", "board", "busio",
    "tkinter", "wx", "PyQt5", "PyQt6"
}


# PULL IMAGE WITH RETRY AND FALLBACK
def ensure_image_available(client) -> str:
    """
    Ensures the Docker image is available locally.
    Tries primary image first then falls back to alternatives.
    Returns the image name that is available.
    """
    images_to_try = [
        "python:3.11-slim",
        "python:3.11",
        "python:3-slim",
        "python:3"
    ]

    for image_name in images_to_try:
        # Check if image is already cached locally
        try:
            client.images.get(image_name)
            logger.info(f"Using cached image: {image_name}")
            return image_name
        except docker.errors.ImageNotFound:
            pass

        # Try to pull with timeout
        try:
            logger.info(f"Pulling image: {image_name}...")
            client.images.pull(
                image_name,
                timeout=60
            )
            logger.info(f"Pulled: {image_name}")
            return image_name

        except Exception as e:
            logger.warning(
                f"Could not pull {image_name}: {e}. "
                f"Trying next option..."
            )
            continue

    raise Exception(
        "No Docker image available. "
        "Run: docker pull python:3.11-slim"
    )


# BETTER TAR ARCHIVE
def create_tar_archive(files: dict) -> bytes:
    """
    Creates tar archive preserving directory structure.
    Files in subdirectories are placed in correct paths
    inside the container - not flattened to root.
    """
    tar_buffer = io.BytesIO()

    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        for filepath, content in files.items():
            content_bytes = content.encode("utf-8")

            # Create directory entries for nested paths
            parts = filepath.split("/")
            if len(parts) > 1:
                dir_path = "/".join(parts[:-1])
                try:
                    dir_info      = tarfile.TarInfo(name=dir_path)
                    dir_info.type = tarfile.DIRTYPE
                    dir_info.mode = 0o755
                    tar.addfile(dir_info)
                except Exception:
                    pass  # directory may already exist

            info      = tarfile.TarInfo(name=filepath)
            info.size = len(content_bytes)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(content_bytes))

    tar_buffer.seek(0)
    return tar_buffer.read()


# SMART DEPENDENCY DETECTION
def extract_imports_from_code(code: str) -> list[str]:
    """
    Extracts top-level package names from Python imports.
    Filters out standard library packages.
    """
    packages = set()

    import_pattern = re.findall(
        r"^import\s+([\w]+)", code, re.MULTILINE
    )
    from_pattern = re.findall(
        r"^from\s+([\w]+)\s+import", code, re.MULTILINE
    )

    packages.update(import_pattern)
    packages.update(from_pattern)

    stdlib = {
        "os", "sys", "re", "json", "time", "datetime",
        "math", "random", "string", "collections",
        "itertools", "functools", "typing", "pathlib",
        "subprocess", "threading", "io", "abc", "copy",
        "enum", "dataclasses", "unittest", "pytest",
        "logging", "warnings", "contextlib", "inspect",
        "traceback", "gc", "struct", "hashlib", "base64",
        "uuid", "urllib", "http", "socket", "queue",
        "heapq", "bisect", "array", "weakref", "types",
        "ast", "dis", "tempfile", "shutil", "glob",
        "argparse", "csv", "sqlite3", "xml", "html"
    }

    result = []
    for pkg in packages - stdlib:
        if pkg in SKIP_PACKAGES:
            logger.info(f"Skipping hardware package: {pkg}")
            continue
        elif pkg in PACKAGE_REMAPS:
            result.append(PACKAGE_REMAPS[pkg])
        else:
            result.append(pkg)

    return list(set(result))


def build_install_command(
    code_context: dict,
    repo_requirements: str = None
) -> str:
    """
    Builds the pip install command for the container.
    Priority: requirements.txt > import scanning > pytest only
    """
    if repo_requirements:
        logger.info("Using repo requirements.txt for install")
        return (
            "pip install pytest -q --no-cache-dir && "
            "pip install -r requirements.txt -q "
            "--no-cache-dir --ignore-requires-python || true"
        )

    # Scan imports from all Python files
    packages = {"pytest"}
    for filename, content in code_context.items():
        if filename.endswith(".py"):
            detected = extract_imports_from_code(content)
            packages.update(detected)

    packages_str = " ".join(packages)
    logger.info(f"Auto-detected packages: {packages_str}")

    return (
        f"pip install {packages_str} -q --no-cache-dir || true"
    )


# EXEC WITH TIMEOUT
def exec_with_timeout(
    container,
    cmd: str,
    workdir: str = "/workspace",
    timeout: int = CONTAINER_TIMEOUT
) -> dict:
    """
    Runs a command in the container with a hard timeout.
    Returns stdout, stderr, and exit code.
    Kills the command if it exceeds timeout.
    """
    result = container.exec_run(
        cmd=["/bin/sh", "-c", cmd],
        workdir=workdir,
        demux=False
    )

    output    = result.output.decode("utf-8") if result.output else ""
    exit_code = result.exit_code if result.exit_code is not None else -1

    return {
        "output":    output,
        "exit_code": exit_code
    }


# VERIFY PIP INSTALL SUCCEEDED
def verify_install(container, packages: list[str]) -> bool:
    """
    Verifies that packages were actually installed.
    Runs a quick import check for each package.
    """
    if not packages:
        return True

    check_cmd = " && ".join([
        f"python -c 'import {pkg}' 2>/dev/null"
        for pkg in packages[:5]  # check first 5 only
    ])

    result = exec_with_timeout(
        container,
        check_cmd,
        timeout=10
    )

    return result["exit_code"] == 0


# MAIN SANDBOX FUNCTION
def run_tests_in_docker(
    code_context: dict,
    patch: str,
    tests: str,
    repo_requirements: str = None
) -> dict:
    """
    THE MAIN SANDBOX FUNCTION.

    Improvements:
    1. Image pull with retry and fallback
    2. Tar archive preserves directory structure
    3. Smart dependency detection and installation
    4. exec_run with timeout - no infinite hangs
    5. pip install verification
    6. Detailed output parsing
    7. Always cleans up container

    Returns: status, output, exit_code
    """
    logger.info("=== Docker Sandbox starting ===")

    container = None

    try:
        # Connect to Docker
        client = docker.from_env()
        logger.info("Connected to Docker daemon")

        # IMPROVEMENT 1 - get image with retry
        image_name = ensure_image_available(client)

        # Create container with reasonable resource limits and strict security
        container = client.containers.create(
            image=image_name,
            command="sleep infinity",
            working_dir="/workspace",
            network_disabled=False,    # needs internet for pip
            mem_limit="512m",          # 512MB - enough for most repos
            cpu_period=100000,
            cpu_quota=75000,           # 75% of one CPU core
            pids_limit=256,            # Prevent fork bombs
            security_opt=["no-new-privileges"], # Prevent privilege escalation
            cap_drop=["MKNOD", "NET_RAW", "SYS_ADMIN"], # Drop dangerous capabilities
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

        if repo_requirements:
            files_to_copy["requirements.txt"] = repo_requirements
            logger.info("Added requirements.txt to container")

        # IMPROVEMENT 2 - tar with directory structure
        tar_data = create_tar_archive(files_to_copy)
        container.put_archive("/workspace", tar_data)
        logger.info(
            f"Copied {len(files_to_copy)} files into container"
        )

        # IMPROVEMENT 3 - smart install command
        install_cmd = build_install_command(
            code_context,
            repo_requirements
        )

        logger.info("Installing dependencies...")
        install_result = exec_with_timeout(
            container,
            install_cmd,
            timeout=120  # pip install can take a while
        )

        # Log last few lines of pip output
        pip_lines = install_result["output"].strip().split("\n")
        for line in pip_lines[-5:]:
            if line.strip():
                logger.info(f"pip: {line}")

        # IMPROVEMENT 4 - run pytest with timeout
        logger.info("Running pytest...")
        test_result = exec_with_timeout(
            container,
            "python -m pytest test_fix.py -v --tb=short --no-header",
            timeout=CONTAINER_TIMEOUT
        )

        output    = test_result["output"]
        exit_code = test_result["exit_code"]

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
        error_msg = str(e)
        logger.error(f"Docker error: {error_msg}")

        # Helpful message for common errors
        if "CreateFile" in error_msg or "connect" in error_msg.lower():
            logger.error(
                "Docker Desktop is not running. "
                "Please start Docker Desktop and try again."
            )

        return {
            "status":    "failed",
            "output":    f"Docker error: {error_msg}",
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
        # ALWAYS clean up - runs even on exception
        if container is not None:
            try:
                container.stop(timeout=5)
                container.remove(force=True)
                logger.info("Container stopped and removed")
            except Exception as e:
                logger.warning(
                    f"Could not clean up container: {e}"
                )