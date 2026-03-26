# sandbox/runner.py

import docker
import tempfile
import os
import tarfile
import io
from utils.logger import get_logger

logger = get_logger(__name__)

# How long we allow the container to run before killing it
CONTAINER_TIMEOUT = 60  # seconds

# The Docker image to use — official Python slim image
# Small, fast to pull, has Python and pip built in
DOCKER_IMAGE = "python:3.11-slim"


def create_tar_archive(files: dict) -> bytes:
    """
    Creates an in-memory tar archive containing multiple files.
    Docker's copy API works with tar archives — not individual files.

    files = {"filename.py": "file content as string"}
    returns bytes of the tar archive
    """
    tar_buffer = io.BytesIO()

    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        for filename, content in files.items():
            # Convert string content to bytes
            content_bytes = content.encode("utf-8")

            # Create a tar info object — metadata about the file
            info = tarfile.TarInfo(name=filename)
            info.size = len(content_bytes)

            # Add the file to the archive
            tar.addfile(info, io.BytesIO(content_bytes))

    # Rewind to start so it can be read
    tar_buffer.seek(0)
    return tar_buffer.read()


def apply_patch_to_code(
    code_context: dict,
    patch: str
) -> dict:
    """
    Applies the unified diff patch to the code context.
    Returns updated dict of filename → modified content.

    For now we use a simple approach:
    We pass both the original files AND the patch to the container
    and let the patch command apply it there.
    This returns the files unchanged — patching happens in container.
    """
    return code_context


def run_tests_in_docker(
    code_context: dict,
    patch: str,
    tests: str
) -> dict:
    """
    THE MAIN SANDBOX FUNCTION.

    Takes:
    - code_context: dict of filename → original code
    - patch: the unified diff from Code Writer
    - tests: the pytest code from Test Writer

    Returns:
    - dict with "status" ("passed"/"failed") and "output" (full logs)
    """
    logger.info("=== Docker Sandbox starting ===")

    try:
        # Connect to Docker daemon running on your machine
        client = docker.from_env()
        logger.info("Connected to Docker daemon")

        # Pull the image if not already present
        logger.info(f"Pulling Docker image: {DOCKER_IMAGE}")
        client.images.pull(DOCKER_IMAGE)

        # Create the container — but don't start it yet
        # network_disabled=True means the container cannot
        # make any network requests — security measure
        container = client.containers.create(
            image=DOCKER_IMAGE,
            command="sleep infinity",  # keeps container alive
            working_dir="/workspace",  # all our files go here
            network_disabled=True,     # no internet access
            mem_limit="256m",          # max 256MB RAM
            cpu_period=100000,
            cpu_quota=50000,           # max 50% of one CPU core
        )

        logger.info(f"Container created: {container.short_id}")

        # Start the container
        container.start()
        logger.info("Container started")

        # Prepare all files to copy into the container
        # We copy: original code files + patch file + test file
        files_to_copy = {}

        # Add all original code files
        for filename, content in code_context.items():
            files_to_copy[filename] = content

        # Add the patch file
        files_to_copy["fix.patch"] = patch

        # Add the test file
        files_to_copy["test_fix.py"] = tests

        # Add a setup script that installs pytest and applies patch
        files_to_copy["setup.sh"] = """#!/bin/bash
pip install pytest --quiet
# Apply the patch if patch command is available
if command -v patch &> /dev/null; then
    patch -p1 < fix.patch || true
fi
"""

        # Create tar archive of all files
        tar_data = create_tar_archive(files_to_copy)

        # Copy the tar archive into the container at /workspace
        container.put_archive("/workspace", tar_data)
        logger.info(
            f"Copied {len(files_to_copy)} files into container"
        )

        # Run setup — install pytest
        logger.info("Installing pytest in container...")
        setup_result = container.exec_run(
            "pip install pytest --quiet --no-cache-dir",
            workdir="/workspace"
        )

        verify_result = container.exec_run(
            "python -m pytest --version",
            workdir="/workspace"
        )
        logger.info(
            f"pytest verify: "
            f"{verify_result.output.decode('utf-8').strip()}"
        )

        # Run the tests
        logger.info("Running pytest in container...")
        test_result = container.exec_run(
            "python -m pytest test_fix.py -v --tb=short",
            workdir="/workspace"
        )

        # Decode the output
        output = test_result.output.decode("utf-8")
        exit_code = test_result.exit_code

        logger.info(f"Tests complete. Exit code: {exit_code}")
        logger.info(f"Test output:\n{output}")

        # pytest exit code 0 means all tests passed
        # any other exit code means tests failed
        if exit_code == 0:
            status = "passed"
            logger.info("Tests PASSED")
        else:
            status = "failed"
            logger.warning("Tests FAILED")

        return {
            "status": status,
            "output": output,
            "exit_code": exit_code
        }

    except docker.errors.DockerException as e:
        logger.error(f"Docker error: {e}")
        return {
            "status": "failed",
            "output": f"Docker error: {str(e)}",
            "exit_code": -1
        }

    except Exception as e:
        logger.error(f"Sandbox error: {e}")
        return {
            "status": "failed",
            "output": f"Sandbox error: {str(e)}",
            "exit_code": -1
        }

    finally:
        # ALWAYS clean up the container
        # This runs even if an exception occurred above
        try:
            container.stop(timeout=5)
            container.remove(force=True)
            logger.info("Container stopped and removed")
        except Exception as e:
            logger.warning(f"Could not clean up container: {e}")