# revolve-manager: dependencies only, no source. The repo is mounted at runtime at its
# host path, with the docker socket (components/manager/src/manager/loop.py). Used for the manager container and
# for every loop step's container (the components' host-side composition functions under revolve_common.wrapper).
FROM python:3.12-slim
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
CMD ["python", "-m", "manager.loop"]
