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

## Host Operations Agent

Start with the public manager health check, then use an existing administrator cookie jar for the
admin-only Agent health endpoint:

```bash
curl -fsS http://127.0.0.1:3000/api/health
curl -fsS -b ./admin.cookies http://127.0.0.1:3000/api/ops-agent/health
```

The second response is intentionally limited to a safe schema: `ok` includes `protocol_version`,
while `unavailable` and `error` include a fixed redacted `detail`. A normal shell request without an
approved Operation Plan returns `approval_required`; this is policy enforcement, not an Agent
outage.

Check socket activation and bounded service logs on the host:

```bash
sudo systemctl status dgx-spark-ops-agent.socket dgx-spark-ops-agent.service --no-pager
sudo journalctl -u dgx-spark-ops-agent.socket -u dgx-spark-ops-agent.service -n 200 --no-pager
sudo stat -c '%U:%G %a %n' \
  /run/dgx-spark-manager/ops-agent.sock \
  /var/lib/dgx-spark-ops-agent/jobs
```

After the Agent has initialized, the expected values are:

```text
root:dgx-spark-ops 660 /run/dgx-spark-manager/ops-agent.sock
root:dgx-spark-ops 640 /etc/dgx-spark-manager/ops-agent.key
root:root          700 /var/lib/dgx-spark-ops-agent/jobs
```

Never make the socket world-writable with `chmod 666`, expose it through TCP, print the key, or paste
raw job metadata into a ticket. Those actions bypass the intended local group boundary or disclose
sensitive diagnostic output.

### Group GID or socket permission mismatch

The Compose group must use the host group's numeric GID. Compare the host value, `.env`, and the
running container:

```bash
getent group dgx-spark-ops
stat -c '%g %a %n' /run/dgx-spark-manager/ops-agent.sock
grep '^OPS_AGENT_GID=' .env
docker compose exec manager id
docker compose exec manager stat -c '%g %a %n' /run/dgx-spark-manager/ops-agent.sock
```

If the values differ, atomically upsert only `OPS_AGENT_GID`. This snippet rejects a missing or
non-numeric group result, keeps every non-target `.env` line in order, writes the temporary file in
the same directory, preserves the existing mode, and does not print `.env` contents:

```bash
set -eu
[ -f .env ] && [ ! -L .env ] || { printf '%s\n' '.env must be a regular file' >&2; exit 1; }
ops_gid="$(getent group dgx-spark-ops | cut -d: -f3)"
case "$ops_gid" in
  ''|*[!0-9]*) printf '%s\n' 'dgx-spark-ops GID is not numeric' >&2; exit 1 ;;
esac
umask 077
tmp="$(mktemp ./.env.ops-agent.XXXXXX)"
trap 'rm -f -- "$tmp"' EXIT HUP INT TERM
{
  found=0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      OPS_AGENT_GID=*)
        if [ "$found" -eq 0 ]; then
          printf 'OPS_AGENT_GID=%s\n' "$ops_gid"
          found=1
        fi
        ;;
      *) printf '%s\n' "$line" ;;
    esac
  done < .env
  if [ "$found" -eq 0 ]; then
    printf 'OPS_AGENT_GID=%s\n' "$ops_gid"
  fi
} > "$tmp"
chmod --reference=.env -- "$tmp"
mv -f -- "$tmp" .env
trap - EXIT HUP INT TERM
docker compose up -d --force-recreate manager
```

If the group and GID match but the socket mode or owner does not, do not loosen its permissions.
Restore the installed unit and let socket activation recreate it:

```bash
sudo ./scripts/install-ops-agent.sh --apply
sudo systemctl restart dgx-spark-ops-agent.socket
sudo systemctl restart dgx-spark-ops-agent.service
```

### Key mismatch or invalid signed response

Do not inspect or modify the key with `cat`, `stat`, `chown`, `chmod`, shell tracing, environment
variables, or ad hoc replacement files. Run the installer, which validates the exact path chain,
regular-file type, absence of symlinks, owner, group, mode, size, and content before using the key.
If it succeeds, its signed probe proves the host Agent and key agree. Recreate the manager to clear
a stale bind mount or cached key:

```bash
sudo ./scripts/install-ops-agent.sh --apply
docker compose up -d --force-recreate manager
curl -fsS -b ./admin.cookies http://127.0.0.1:3000/api/ops-agent/health
```

If any key path, type, symlink, owner, group, mode, size, or content check fails, the installer fails
closed and does not repair or overwrite the key. Do not change the key inode manually. Before
rotating it, confirm there are no active Agent jobs, accept a brief Agent outage, and create a
current manager database and `.env` backup. Then use the confirmation-gated purge contract,
reinstall, and recreate the manager:

```bash
./scripts/backup.sh
# Type PURGE OPS AGENT KEY when prompted.
sudo ./scripts/uninstall-ops-agent.sh --apply --purge-key
sudo ./scripts/install-ops-agent.sh --apply
docker compose up -d --force-recreate manager
curl -fsS -b ./admin.cookies http://127.0.0.1:3000/api/ops-agent/health
```

This uninstaller contract removes the Agent package and units as well as the invalid key, but
preserves the job directory because `--purge-jobs` was not supplied. Reinstallation generates a new
key, and manager recreation is required before authenticated Agent requests can resume. The
uninstaller also fails closed if the key path itself is unsafe; if it refuses the purge, stop rather
than bypassing its path checks with manual file operations.

### Agent jobs, timeouts, and restart recovery

Agent job metadata and bounded redacted output are stored together at:

```text
/var/lib/dgx-spark-ops-agent/jobs/<job-id>.json
```

The directory is root-only and files are mode `0600`. Inspect status through the panel or manager
API first. If a job remains running after its configured timeout, restart the Agent once and check
recovery plus service logs:

```bash
sudo systemctl restart dgx-spark-ops-agent.service
sudo journalctl -u dgx-spark-ops-agent.service -n 200 --no-pager
sudo find /var/lib/dgx-spark-ops-agent/jobs -maxdepth 1 -type f \
  -name '*.json' -printf '%u:%g %m %f\n'
```

Restart recovery validates process identity before signalling interrupted work; systemd
`KillMode=control-group` contains remaining service processes. Do not manually kill a PID copied
from a metadata file because PID reuse can target an unrelated process.

Structured read tools may use up to 15 seconds, followed by bounded termination and completion
grace. Keep `DGX_OPS_AGENT_READ_TIMEOUT_SECONDS` at the default 30 seconds or above the complete
server-side budget. A lower client timeout can report `unavailable` while the Agent is still safely
finishing or cancelling the read job. After changing it, recreate the manager container.

### Reinstall, rollback, and physical purge

Preview both lifecycle operations before changing the host:

```bash
./scripts/install-ops-agent.sh
./scripts/uninstall-ops-agent.sh
```

Install apply is transactional and preserves the key and job history. A failed apply attempts to
restore the previous package, units, and systemd state. Normal uninstall also preserves the key and
jobs so rollback or reinstall retains authentication and diagnostics:

```bash
sudo ./scripts/uninstall-ops-agent.sh --apply
sudo ./scripts/install-ops-agent.sh --apply
```

Only use the purge flags when physical deletion is intended. Each flag requires a separate exact
interactive confirmation:

```bash
# Type PURGE OPS AGENT JOBS when prompted.
sudo ./scripts/uninstall-ops-agent.sh --apply --purge-jobs
# Type PURGE OPS AGENT KEY when prompted.
sudo ./scripts/uninstall-ops-agent.sh --apply --purge-key
```

Purging the key requires recreating the manager after reinstall so it mounts the newly generated
key. Purging jobs permanently deletes their metadata and retained redacted output.

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
