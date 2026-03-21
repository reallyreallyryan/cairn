"""Docker-based sandbox for isolated code execution.

Uses Colima/Docker to run Python code in isolated containers with
resource limits, no network access, and restricted filesystem.
"""

import io
import logging
import os
import tarfile
from pathlib import Path

import yaml

from config.settings import settings

logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages Docker containers for sandboxed code execution."""

    def __init__(self, policy_path: str | None = None):
        import docker

        self._docker_host = self._resolve_docker_host()
        self._policy = self._load_policy(
            policy_path or settings.sandbox_policy_path
        )
        self._image = self._policy.get("sandbox", {}).get(
            "image", settings.sandbox_image
        )

        try:
            self._client = docker.DockerClient(
                base_url=self._docker_host
            )
            self._client.ping()
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Docker at {self._docker_host}. "
                f"Is Colima running? Try: colima start\nError: {e}"
            )

    def _resolve_docker_host(self) -> str:
        """Resolve Docker socket path with fallback chain."""
        # 1. Explicit setting
        if settings.docker_host:
            return settings.docker_host

        # 2. DOCKER_HOST env var
        env_host = os.environ.get("DOCKER_HOST")
        if env_host:
            return env_host

        # 3. Colima default socket
        colima_socket = Path.home() / ".colima" / "default" / "docker.sock"
        if colima_socket.exists():
            return f"unix://{colima_socket}"

        # 4. Default Docker socket
        default_socket = Path("/var/run/docker.sock")
        if default_socket.exists():
            return f"unix://{default_socket}"

        return "unix:///var/run/docker.sock"

    def _load_policy(self, path: str) -> dict:
        """Load sandbox policy from YAML file."""
        policy_path = Path(path)
        if not policy_path.exists():
            logger.warning("Sandbox policy not found at %s, using defaults", path)
            return {"sandbox": {
                "resource_limits": {"memory": "256m", "timeout_seconds": 60, "max_output_chars": 10000},
                "network": {"enabled": False},
                "filesystem": {"read_only_root": True},
            }}

        with open(policy_path) as f:
            return yaml.safe_load(f)

    def _get_resource_config(self) -> dict:
        """Extract Docker container config from policy."""
        limits = self._policy.get("sandbox", {}).get("resource_limits", {})
        network = self._policy.get("sandbox", {}).get("network", {})
        fs = self._policy.get("sandbox", {}).get("filesystem", {})

        config = {
            "mem_limit": limits.get("memory", "256m"),
            "network_disabled": not network.get("enabled", False),
        }

        # CPU limits
        cpu_period = limits.get("cpu_period")
        cpu_quota = limits.get("cpu_quota")
        if cpu_period and cpu_quota:
            config["cpu_period"] = cpu_period
            config["cpu_quota"] = cpu_quota

        # Read-only root filesystem with writable tmpfs mounts
        if fs.get("read_only_root", True):
            config["read_only"] = True
            tmpfs = {}
            for path in fs.get("writable_paths", ["/sandbox/workspace", "/sandbox/output", "/tmp"]):
                tmpfs[path] = "size=50m"
            config["tmpfs"] = tmpfs

        return config

    def _make_tar(self, filename: str, content: str) -> bytes:
        """Create a tar archive containing a single file."""
        encoded = content.encode("utf-8")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=filename)
            info.size = len(encoded)
            tar.addfile(info, io.BytesIO(encoded))
        buf.seek(0)
        return buf.read()

    def execute_code(self, code: str, timeout: int | None = None) -> dict:
        """Execute Python code in an isolated Docker container.

        Returns dict with: stdout, stderr, exit_code, timed_out, output_files
        """
        limits = self._policy.get("sandbox", {}).get("resource_limits", {})
        timeout = timeout or limits.get("timeout_seconds", 60)
        max_output = limits.get("max_output_chars", 10000)

        container = None
        try:
            # Create container without read_only so we can inject code,
            # then rely on non-root user + tmpfs for security
            resource_config = self._get_resource_config()
            resource_config.pop("read_only", None)

            # Pass code via python3 -c to avoid file injection issues with tmpfs
            container = self._client.containers.create(
                image=self._image,
                command=["python3", "-c", code],
                detach=True,
                **resource_config,
            )

            # Start and wait
            container.start()
            result = container.wait(timeout=timeout)

            # Capture output
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

            # Truncate
            if len(stdout) > max_output:
                stdout = stdout[:max_output] + f"\n[Truncated at {max_output} chars]"
            if len(stderr) > max_output:
                stderr = stderr[:max_output] + f"\n[Truncated at {max_output} chars]"

            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result.get("StatusCode", -1),
                "timed_out": False,
                "output_files": [],
            }

        except Exception as e:
            error_str = str(e)
            timed_out = "timed out" in error_str.lower() or "read timeout" in error_str.lower()

            if timed_out and container:
                try:
                    container.kill()
                except Exception:
                    pass

            if timed_out:
                return {
                    "stdout": "",
                    "stderr": f"Execution timed out after {timeout} seconds",
                    "exit_code": -1,
                    "timed_out": True,
                    "output_files": [],
                }

            logger.error("Sandbox execution error: %s", e)
            return {
                "stdout": "",
                "stderr": f"Sandbox error: {e}",
                "exit_code": -1,
                "timed_out": False,
                "output_files": [],
            }

        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception as e:
                    logger.warning("Failed to remove container: %s", e)

    def execute_file(self, file_path: str, timeout: int | None = None) -> dict:
        """Execute a Python file in the sandbox."""
        path = Path(file_path)
        if not path.exists():
            return {"stdout": "", "stderr": f"File not found: {file_path}",
                    "exit_code": -1, "timed_out": False, "output_files": []}
        code = path.read_text(encoding="utf-8")
        return self.execute_code(code, timeout=timeout)

    def check_health(self) -> dict:
        """Verify Docker connection and sandbox image availability."""
        try:
            self._client.ping()
            docker_ok = True
        except Exception as e:
            return {"docker": False, "image": False, "error": str(e)}

        try:
            self._client.images.get(self._image)
            image_ok = True
        except Exception:
            image_ok = False

        return {
            "docker": docker_ok,
            "image": image_ok,
            "socket": self._docker_host,
            "image_name": self._image,
            "build_hint": f"docker build --platform linux/arm64 -t {self._image} sandbox/"
            if not image_ok else None,
        }

    def cleanup_all(self):
        """Remove all cairn sandbox containers (orphan cleanup)."""
        try:
            containers = self._client.containers.list(
                all=True,
                filters={"ancestor": self._image},
            )
            for c in containers:
                c.remove(force=True)
                logger.info("Cleaned up container: %s", c.short_id)
        except Exception as e:
            logger.warning("Cleanup failed: %s", e)
