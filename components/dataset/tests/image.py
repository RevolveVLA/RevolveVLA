"""Test-only access to the dataset image and its job containers, built from launch's own pieces (docker_cmd for the
user, environment and mounts, container for the image with Rosetta sourced), so a test command sees exactly what the
entrypoint sees.

  r = in_image(launch, cmd, mounts_for=["merge", "--out", D, DATASET])    any command in the image
  ip = job_ip(job)                                                        a launch.start job's container address"""
from __future__ import annotations

import subprocess


def in_image(launch, cmd, mounts_for=(), timeout=1800) -> subprocess.CompletedProcess:
    """cmd in the dataset image, removed when it exits; mounts_for: entrypoint-style arguments whose paths get mounted
    as the entrypoint would get them."""
    opts, _ = launch.docker_cmd(list(mounts_for))
    return subprocess.run([*opts, "--rm", *launch.container(list(cmd))], capture_output=True, text=True,
                          timeout=timeout)


def job_ip(job) -> str:
    """The job container's IP address, where the HTTP job wrapper listens on port 8000."""
    return subprocess.run(["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                           job.container], capture_output=True, text=True, check=True).stdout.strip()
