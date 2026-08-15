# Troubleshooting

## Manager Does Not Start

```bash
docker compose ps
docker compose logs --tail=200 manager
docker inspect dgx-spark-web-manager --format '{{json .State.Health}}'
```

Verify `.env` exists, `DGX_SECRET_KEY` is at least 32 characters, and `DGX_ADMIN_PASSWORD` is at least 12 characters.

## Docker Permission Denied

The manager runs as the host user and adds the Docker socket group dynamically. Refresh it and recreate:

```bash
sed -i "s/^DOCKER_GID=.*/DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)/" .env
docker compose up -d --force-recreate
```

## GPU Or `nvidia-smi` Missing

Confirm the host and a minimal container both see the GPU:

```bash
nvidia-smi
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

Then verify `NVIDIA_VISIBLE_DEVICES=all` and `NVIDIA_DRIVER_CAPABILITIES=compute,utility` in `docker compose config`.

## Existing Models Not Found

Check `.env` paths and container mounts:

```bash
grep -E 'HF_HOME_HOST|MODEL_HOME_HOST' .env
docker compose exec manager ls -la /hf-cache/hub /models
```

Use **系统概览 -> 重新发现** after correcting mounts. Scanning never moves or deletes unknown files.

## Download Fails

- Configure a Hugging Face Token in **系统设置** for gated/private repositories.
- Check free disk space under the host HF cache.
- Open **任务中心**, inspect the error and resume. Completed Hub blobs are retained.
- Confirm the repository uses `owner/model` and that access terms were accepted on Hugging Face.

## Deployment Does Not Become Healthy

The manager waits up to 120 seconds, captures failure in the task, stops the new container, and removes it. Existing containers are untouched.

Check:

```bash
docker ps -a --filter label=com.dgx-spark-manager.managed=true
docker logs <container>
free -h
nvidia-smi
```

Reduce memory fraction, context length, or concurrency. DGX Spark reports GPU memory as unsupported because CPU and GPU share unified memory; use host memory availability instead.

## OpenAI SDK Returns 401

Create a gateway key under **API 网关**. Administrator passwords and browser cookies are not valid gateway keys. A revoked key cannot be restored.

## Backup And Recovery

Run `./scripts/backup.sh` before upgrades. Model files are not copied because they remain in the host model/HF directories. `restore.sh` validates archive paths, stops the manager, restores SQLite and `.env`, then restarts the service.
