# sandbox/runner.py

import docker
import tempfile
import os
import tarfile
import io
from utils.logger import get_logger

logger = get_logger(__name__)

CONTAINER_TIMEOUT = 60
DOCKER_IMAGE = "python:3.11-slim"


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
    tests: str
) -> dict:
    logger.info("=== Docker Sandbox starting ===")

    try:
        client = docker.from_env()
        logger.info("Connected to Docker daemon")

        logger.info(f"Pulling Docker image: {DOCKER_IMAGE}")
        client.images.pull(DOCKER_IMAGE)

        # network_disabled=False so pip can install pytest
        container = client.containers.create(
            image=DOCKER_IMAGE,
            command="sleep infinity",
            working_dir="/workspace",
            network_disabled=False,
            mem_limit="256m",
            cpu_period=100000,
            cpu_quota=50000,
        )

        logger.info(f"Container created: {container.short_id}")

        container.start()
        logger.info("Container started")

        files_to_copy = {}

        for filename, content in code_context.items():
            files_to_copy[filename] = content

        files_to_copy["fix.patch"] = patch
        files_to_copy["test_fix.py"] = tests

        tar_data = create_tar_archive(files_to_copy)

        container.put_archive("/workspace", tar_data)
        logger.info(
            f"Copied {len(files_to_copy)} files into container"
        )

        # Install pytest and run tests in one single command
        # This ensures pytest is available when tests run
        logger.info("Installing pytest and running tests...")
        test_result = container.exec_run(
            cmd=[
                "/bin/sh", "-c",
                "pip install pytest -q && "
                "python -m pytest test_fix.py -v --tb=short"
            ],
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