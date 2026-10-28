```text
     ┌───────┐
     │ ◉   ◉ │       C   L   A   U   D   E
     │   ▿   │
     └─┬───┬─┘       NYU Torch cluster: access, GPUs, Slurm rules,
     ┌─┴───┴─┐       storage, OSN pod and software environment.
     │ ▓▓▓▓▓ │
     └─┘   └─┘
```

# NYU TORCH HPC

**Last updated: 2026-10-27**

> **Keep this file up to date.** If the date above is **more than one month old**, before relying on this file:
> 1. launch an agent to re-read the [NYU RTS docs](https://services.rt.nyu.edu/docs/hpc/getting_started/intro/)
>    (source repo [NYU-RTS/rts-docs](https://github.com/NYU-RTS/rts-docs)) and check that nothing changed;
> 2. check your memory for anything useful learned on Torch since then;
> 3. update this file, its date, and tell me what changed.

Distilled from the NYU RTS docs, checked on the cluster on 2026-10-14, completed with what we learned while building
the datasets and training the encoders of Heimdall (2026-10-22 to 2026-10-27).
**[verify]** = not in the docs or ambiguous: check before relying on it.

**Contents**

0. In short
1. Access
2. Hardware
3. Our resources
4. Slurm
5. Storage
6. Big datasets
7. Data transfers
8. OSN pod
9. Software environment
10. Multi-node DDP template
11. Misc
12. Open questions

---

## 0. IN SHORT

1. **`--account` is mandatory, `--partition` is forbidden**: pick the GPU with `--constraint` (`h200`, `rtx6000`, …).
2. **`--mem=0` and `--exclude` are refused**: ask for the memory explicitly, report a bad node to hpc@nyu.edu.
3. **16 GPUs per user for jobs ≤ 48 h, 4 GPUs beyond**, counted over all the running jobs.
4. **Underused GPUs are cancelled**: below 60 % on H200, 50 % on RTX / L40S / A100.
5. **The lab has 8 RTX in total** (`lzanna`): big trainings go on `courant` (H200, L40S).
6. **Scratch is purged after 60 days without access**; home has only 30 K files.
7. **No transfer on login nodes** (DTN or Globus), no heavy work either (2 GB of memory per user).
8. **Jobs mount the environment overlay read-only** (`:ro`); install packages on a compute node.

---

## 1. ACCESS

| Item          | Details                                                                                                                                                                         |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Requirements  | an *HPC account* (NetID, gives the login) **and** an *active allocation* on https://projects.hpc.nyu.edu (gives a Slurm account `torch_pr_XXX_XXXXX`). No allocation → no jobs. |
| Collaborators | a full-time NYU faculty sponsors an *affiliate* NetID, then the person requests the HPC account ("External Collaborator"). Expires after **12 months** → yearly renewal.        |
| VPN           | **mandatory** off campus (login, OOD, project portal, status dashboard). Linux: `sudo openconnect -b vpn.nyu.edu`.                                                              |
| Login         | **no SSH keys**: password + Microsoft device-login 2FA (https://microsoft.com/devicelogin + Duo) ⇒ no non-interactive SSH from outside, automation runs *on* Torch.             |
| VSCode        | `code-server` app in Open OnDemand (https://ood.torch.hpc.nyu.edu), or Remote-SSH (settings below).                                                                             |
| Cluster usage | https://stats.apps.cloud.rt.nyu.edu (VPN), or `vsinfo` (in `~/bin`).                                                                                                            |
| Support       | hpc@nyu.edu (personal consultations, including deep-learning scaling).                                                                                                          |

**Good to know**

- VSCode Remote-SSH: `Remote.SSH: Connect Timeout = 120`, "Lockfiles In Tmp" on, "Use Flock" off, and
  `~/.vscode-server` symlinked to `$SCRATCH/.vscode-server` (home inode quota).
- A reconnection can land on another login node (`torch-login-a-0`, `a-1`, `b-1`, `b-3`, …): processes started from
  the previous one keep running there (`ssh torch-login-b-3` from Torch to stop them).

`~/.ssh/config` (from the docs):

```text
Host torch login.torch.hpc.nyu.edu
  Hostname login.torch.hpc.nyu.edu
  User <NetID>
  StrictHostKeyChecking no
  ServerAliveInterval 60
  ForwardAgent yes
  UserKnownHostsFile /dev/null
  LogLevel ERROR
Host dtn.torch.hpc.nyu.edu
  User <NetID>
  (same options)
Host cs* cl* gr* ga* gh* gl*          # compute nodes, through the login node
  User <NetID>
  ProxyCommand ssh -W %h:%p torch-login
  ForwardAgent yes
```

---

## 2. HARDWARE

Infiniband NDR400. 4 login nodes, 2 DTN nodes (100 Gb/s to the research network).

| Type          | Prefix (`sinfo`)  | Nodes | GPUs / node | GPU memory (PyTorch) | CPUs / node |    RAM / node | Total GPUs |
| ------------- | ----------------- | ----: | ----------: | -------------------: | ----------: | ------------: | ---------: |
| H200          | `gh1*`            |    34 |           8 |               150 GB |         128 |          2 TB |        272 |
| L40S          | `gl*`             |    68 |           4 |                    – |         128 |        512 GB |        272 |
| B200          | `gb*`             |    15 |           8 |               191 GB |         128 |          3 TB |        120 |
| RTX Pro 6000  | `gr*` (gr101-106) |     6 |           8 |               102 GB |         128 |        1.5 TB |         48 |
| H100          | `gh0*`            |    15 |           4 |                    – |          96 |        1.5 TB |         60 |
| A100          | `ga*`             |  9+34 |           4 |                    – |     64 / 80 |    0.5 / 1 TB |        172 |
| CPU std / big | `cs*` / `cl*`     | 186/8 |           – |                    – |         128 | 512 GB / 3 TB |          – |

**Good to know**

- **RTX Pro 6000** = Blackwell (driver 610, sm_120) ⇒ PyTorch built for **CUDA ≥ 12.8**. No NVLink (PCIe only) ⇒
  intra-node all-reduce slower than on H200 / B200 (general knowledge, not from the docs).
- H200 node: `RealMemory=2061000` MB (`scontrol show node gh110`).
- GPU memory = `torch.cuda.get_device_properties(0).total_memory`, measured on 2026-10-25.

---

## 3. OUR RESOURCES

Project **"Zanna Research"** (Laure Zanna, M²LInES) on https://projects.hpc.nyu.edu (verified 2026-10-14).

### Allocations

| Allocation        | Slurm account          | Reaches (no `--constraint`)                 | Ends                   |
| ----------------- | ---------------------- | ------------------------------------------- | ---------------------- |
| `torch` (Cluster) | `torch_pr_347_general` | CPUs, `l40s_public`, `h200_public`          | 2035-06-01             |
| `lzanna`          | `torch_pr_347_lzanna`  | `rtx6000_lzanna` + public GPUs              | 2027-07-14             |
| `courant`         | `torch_pr_347_courant` | `*_courant` (L40S, H200, RTX) + public GPUs | **2026-12-10** ⚠ renew |

### Group caps

Shared by **every member** of the group (`sacctmgr show qos`).

| Partition         |  GPUs | CPUs |  Memory | Shared by                      |
| ----------------- | ----: | ---: | ------: | ------------------------------ |
| `rtx6000_lzanna`  | **8** |  128 |  1400 G | the whole lab (= one RTX node) |
| `rtx6000_courant` |     8 |  128 |  1400 G | all of Courant                 |
| `l40s_courant`    |    52 | 1664 |  6500 G | all of Courant                 |
| `h200_courant`    |    88 | 1408 | 22000 G | all of Courant                 |
| `l40s_public`     |   208 | 6656 | 26000 G | everyone                       |
| `h200_public`     |    24 |  384 |  6000 G | everyone                       |

### Which account and constraint

Verified with held jobs.

| Need                 | Flags                                                | Lands on                         |
| -------------------- | ---------------------------------------------------- | -------------------------------- |
| Lab RTX only         | `--account=torch_pr_347_lzanna --constraint=rtx6000` | `rtx6000_lzanna`                 |
| Courant H200         | `--account=torch_pr_347_courant --constraint=h200`   | `h200_courant` (+ `h200_public`) |
| Courant L40S         | `--account=torch_pr_347_courant --constraint=l40s`   | `l40s_courant` (+ `l40s_public`) |
| Any GPU we can reach | `--account=torch_pr_347_courant` (no constraint)     | first free partition             |
| CPU only             | `--account=torch_pr_347_general`                     | `cs`, `cpu_short`                |

**Good to know**

- CPU-only jobs land on the CPU partitions whatever the account ⇒ use `general` for them.
- 8 RTX on `lzanna` = **all** the lab's GPUs: keep big trainings on `courant`, and `lzanna` for jobs needing the RTX
  or a guaranteed slot.

---

## 4. SLURM

### Rules

| Rule                                  | Details                                                                                                     |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `--account=<slurm_account>` mandatory | on every `srun` / `sbatch`.                                                                                 |
| **Never `--partition`**               | partitions are QoS views: Slurm dispatches to every partition the account reaches that matches the request. |
| GPU type with `--constraint`          | `rtx6000`, `l40s`, `h200`, `h100`, `a100`, `b200`, `cuda`, or `'h200\|l40s'` (`sinfo -o "%P %f"`).          |
| `--mem=0` refused                     | "Please do not request memory 0": ask explicitly, e.g. `--mem=1400G` per RTX node.                          |
| `--exclude=<node>` refused            | "Please do note --exclude … report back to hpc@nyu.edu": report the node instead.                           |

### Limits per user

`show_slurm_qos`. The QoS follows the wall time whatever the account; the group caps (§ 3) apply on top.

| QoS           | Wall time | Max per user          | Used when                       |
| ------------- | --------- | --------------------- | ------------------------------- |
| `interactive` | 6 h       | 16 CPUs, 60 GB        |                                 |
| `cpu_short`   | 6 h       | 32 CPUs, 120 GB       |                                 |
| `cpu48`       | 2 d       | 3000 CPUs, 6000 GB    | CPU jobs ≤ 48 h                 |
| `cpu168`      | 7 d       | 1000 CPUs, 2000 GB    | CPU jobs > 48 h                 |
| `cpuprem`     | 2 d       | 30000 CPUs (preempt.) |                                 |
| **`gpu48`**   | **2 d**   | **16 GPUs**           | GPU jobs ≤ 48 h                 |
| **`gpu168`**  | **7 d**   | **4 GPUs**            | GPU jobs > 48 h (sbatch notice) |

**Good to know**

- GPU caps are **per user, over all running jobs** (`MaxTRESPerUser`, no per-job cap): one job of 16 GPUs, or 2 of 8,
  the next ones waiting (`QOSMaxGRESPerUser`).
- Queue: at most **500** (`gpu48`) or **50** (`gpu168`) GPU jobs, and **3000** CPU jobs (`cpu48`), **each task of an
  array counting** ⇒ submit big arrays in rolling batches.
- The docs mention "24 GPUs per user for jobs < 48 h", but `show_slurm_qos` gives `gres/gpu=16` (2026-10-14).
- Multi-node trainings: jobs ≤ 48 h, chained with checkpoint/restart.
- A busy cluster starts array tasks in slow batches (~15 min apart): `JobArrayTaskLimit` with no task running is slow,
  not stuck.

### Cancellation of underused GPUs

Cancelled automatically ("Enforcement will be very aggressive"), after a delay not documented **[verify]**.

| Nodes                                 | Cancelled below | Warned below |
| ------------------------------------- | --------------: | -----------: |
| `gh*` (H200)                          |            60 % |         75 % |
| `gl*`, `ga*`, `gr*` (L40S, A100, RTX) |            50 % |         70 % |
| other                                 |            10 % |         50 % |

⇒ Long CPU-only phases in GPU jobs (data staging, validation on CPU, big evaluations) can get the job killed: keep the
GPU busy, stage the data in a CPU job beforehand, run notebooks on CPUs.

### Preemption

| Partitions                                                                 | Preempted?                       |
| -------------------------------------------------------------------------- | -------------------------------- |
| `*_lzanna`, `*_courant`, `*_public`, `cs`, `cpu_short` (`PreemptMode=OFF`) | **never**                        |
| `h200`, `l40s`, `rtx6000`, `b200_courant_prem`, … (with `preemption=yes`)  | yes, after 30 min, then requeued |

| `#SBATCH --comment=...`                                        | Effect                                                |
| -------------------------------------------------------------- | ----------------------------------------------------- |
| `"preemption=yes;requeue=true"`                                | normal + preemptible partitions (idle GPUs of others) |
| `"preemption=yes;preemption_partitions_only=yes;requeue=true"` | preemptible only (may allow more resources)           |
| `gpu_mps=yes`                                                  | several jobs share a GPU                              |
| `ram_disk=1GB`                                                 | RAM-backed directory `$SLURM_RAM_TMPDIR`              |

**Good to know**

- Priority: stakeholder > public > preemptible partitions; GPU jobs preempt CPU jobs on GPU nodes.
- Requeued jobs restart from scratch ⇒ the training script must resume from its last checkpoint by itself.
- **B200 (191 GB)** are only reachable this way, e.g. `--account=torch_pr_347_courant --constraint=b200` →
  `b200_courant_prem` (12 nodes × 8 B200, open to every account): handy for short memory tests.

### Pending reasons

| Reason                                                              | Meaning                                                      |
| ------------------------------------------------------------------- | ------------------------------------------------------------ |
| `QOSGrpGRES`                                                        | no free GPU in the partition (e.g. the 8 lab RTX all in use) |
| `QOSMaxGRESPerUser`, `QOSMaxMemoryPerUser`, `QOSMaxCpuPerUserLimit` | your personal limit                                          |

### Useful commands

| Need                                | Command                                                                        |
| ----------------------------------- | ------------------------------------------------------------------------------ |
| Start estimate, without submitting  | `sbatch --test-only job.sbatch` (also with other `--constraint`, `--gres`)     |
| GPU use of a running job            | `srun --jobid=<id> --overlap -n1 nvidia-smi` (array task: id from `squeue %A`) |
| Override the resources of an sbatch | `sbatch --array=5 --ntasks-per-node=2 --gres=gpu:2 --mem=128G job.sbatch`      |
| A whole H200 node                   | `--gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=16 --mem=2000G`              |
| Short memory test on a big GPU      | `--constraint=b200 --comment="preemption=yes;preemption_partitions_only=yes"`  |
| Accounts, quotas, QoS               | `my_slurm_accounts`, `myquota`, `rpsquota`, `show_slurm_qos`                   |
| Efficiency of a finished job        | `seff <jobid>`                                                                 |
| GPU monitoring                      | `nvidia-smi -l 5`, `/share/apps/images/run-nvtop-3.2.0.bash nvtop`             |

**Good to know**

- A **whole H200 node** can wait **1–2 days** on `h200_courant`; partial requests start a bit sooner but run slower.
- Cancelling a pending job loses its place in the queue.
- Don't `watch squeue` / `sacct` (RPC throttling, `sdiag`); use `--mail-type` and job arrays.

---

## 5. STORAGE

| Path                 | Variable                                   | Quota (space / files) | Backup       | Purge             | Use                 |
| -------------------- | ------------------------------------------ | --------------------- | ------------ | ----------------- | ------------------- |
| `/home/$USER`        | `$HOME`                                    | 50 GB / **30 K**      | daily        | no                | code, configs       |
| `/scratch/$USER`     | `$SCRATCH`                                 | 5 TB / 5 M            | **no**       | **unused ≥ 60 d** | jobs, all-flash     |
| `/archive/$USER`     | `$ARCHIVE`                                 | 2 TB / 20 K           | daily        | no                | tar, **login only** |
| RPS (`/projects/…`?) | –                                          | paid                  | daily (30 d) | no                | shared lab space    |
| Node-local disk      | `$SLURM_TMPDIR`, `/state/partition1/$USER` | –                     | no           | end of job        | fast job I/O        |

**Good to know**

- Faking access times (`touch` scripts) to dodge the purge = **account locked**.
- **Home inode trap**: no conda environments in `$HOME`; symlink `.vscode-server`, `.cursor-server` and conda `pkgs`
  to the scratch.
- RPS: $100/TB/yr + $100 per 200 K inodes/yr, at least 1 TB + 200 K inodes, requested by the PI by email. Does the
  Zanna group have one? (stakeholder list: Google sheet linked in the docs).
- Sharing: NFSv4 ACLs (`nfs4_setfacl -a "A::<NetID>:RX" dir`, `-R -a "A:df:<NetID>:RWX" dir/share`), or an IPA group
  `g-DEPT-LAB` (ask hpc@nyu.edu). Never `chmod 777`.
- High-risk data (PII, PHI, CUI) is forbidden on HPC.

---

## 6. BIG DATASETS

- **Avoid millions of small files** (inode quotas, metadata bottleneck): a few big files (Zarr with large chunks, HDF5,
  LMDB) or a **squashfs image** mounted read-only, shareable by several jobs:
  ```bash
  mksquashfs mydata mydata.sqf -keep-as-directory      # build on node-local disk
  apptainer exec --overlay mydata.sqf:ro …             # data appears at /mydata
  ```
- Fastest reads: stage a subset to `$SLURM_TMPDIR` or a RAM disk at the start of the job (mind the GPU utilization
  killer while staging).
- **Internet on compute nodes works**, despite the docs: `api.wandb.ai`, PyPI, GitHub and `nyu1.osn.mghpcc.org`
  (2026-10-14).
- Public datasets (read-only): `/projects/work/public/ml-datasets/`.

---

## 7. DATA TRANSFERS

| Way                         | When                             |
| --------------------------- | -------------------------------- |
| DTN `dtn.torch.hpc.nyu.edu` | rsync, scp, rclone               |
| **Globus** (`nyu#torch`)    | large volumes, automatic retries |
| Open OnDemand upload        | small files only                 |

- **Never transfer on login nodes.**
- `rclone`: on the DTN and as a module (`module load rclone/1.72.1`, the only version on 2026-10-22).

---

## 8. OSN POD

Open Storage Network, S3-compatible object storage of M²LInES.

| Item       | Details                                                                                                                              |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Endpoint   | **`https://nyu1.osn.mghpcc.org`**                                                                                                    |
| Buckets    | **`m2lines-pubs`**, public: anonymous listing and reads since 2026-10-22. Others exist, e.g. `emulators`.                            |
| Heimdall   | datasets 2000-2025 in **`m2lines-pubs/Datasets`** (~12.4 TB, `README.md` at its root), paths in `heimdall/config`, read anonymously. |
| Writing    | only with the rclone remote **`osn-rw`** or the s3fs profile **`osn`**, set up privately by each user.                               |
| Access     | *open* buckets: anonymous read, keys to write; *protected*: keys for both, **two pairs RO / RW** → RO in jobs.                       |
| Naming     | lowercase, digits, `-_.`; no names differing only by case.                                                                           |
| Keys, help | https://coldfront.osn.mghpcc.org (keys), https://openstoragenetwork.github.io/docs/, help@osn.mghpcc.org                             |

Read from Python (public bucket):

```python
ds = xr.open_zarr("s3://m2lines-pubs/Datasets/Glorys/cmems_glorys_2000-2025_daily_4320x2041x50.zarr",
                  storage_options={"anon": True, "client_kwargs": {"endpoint_url": "https://nyu1.osn.mghpcc.org"}})
```

rclone remote (written privately, keys never in a repository nor displayed), then `rclone ls osn:/m2lines-pubs` or
`rclone copy --transfers=32 osn:/bucket/path $SCRATCH/path` (on the DTN):

```text
[osn]
type = s3
provider = Ceph
access_key_id = <key>
secret_access_key = <secret>
endpoint = https://nyu1.osn.mghpcc.org
no_check_bucket = true
```

**Good to know**

- PyTorch `DataLoader` reading OSN with s3fs: **spawned** workers (`multiprocessing_context="spawn"`), forked ones hang.
- ~30 MB/s per S3 stream ⇒ speed comes from the dataloader workers, not the batch size; sharded zarr (one object per
  day) is ~2× faster to read than one object per chunk.
- `rclone about` is not supported; `rclone purge` prints a harmless "Failed to read versioning status". Listings
  (`rclone lsf`, `rclone size --fast-list`) are fine on a login node; big copies go on the DTN or in a job.

---

## 9. SOFTWARE ENVIRONMENT

Apptainer + overlay: the environment lives in a single file ⇒ no inode problem.

| Heimdall | Value                                                                                                                                                                                                              |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Overlay  | `$SCRATCH/env/heimdall.ext3` (15 GB, 500 K files)                                                                                                                                                                  |
| Image    | `/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif`                                                                                                                                                     |
| Env      | conda environment `heimdall` (Miniforge), PyTorch `cu130`                                                                                                                                                          |
| Build    | `sbatch cluster/jobs/environment.sbatch` creates or updates everything                                                                                                                                             |
| Jupyter  | `sbatch cluster/jobs/jupyter.sbatch` (add `--mem=96G` for a full day of GLORYS), then VSCode *Select Kernel → Existing Jupyter Server* with `http://<node>:8888/lab?token=…` (token in `cluster/logs/jupyter.log`) |

**Good to know**

- No Docker, Apptainer only (Docker images work: `apptainer pull docker://…`). Add `--nv` for GPUs.
- **Install packages on a compute node**: login nodes cap memory at **2 GB/user** and run a *different OS*
  (`srun --account=<acct> -c4 --mem=16GB -t2:00:00 --pty bash`).
- **Jobs mount the overlay `:ro`**: a `:rw` mount locks the file (no other job may use it meanwhile).
- `/tmp` is local to each node: files written there on a login node are not visible from the compute nodes.
- No `conda init` block in `~/.bashrc` (it breaks overlays). Use `export PYTHONNOUSERSITE=True`.
- Images: `ls /share/apps/images/`; overlay templates: `ls /share/apps/overlay-fs-ext3/`; Lmod modules (`module avail`),
  e.g. `anaconda3/2024.02`, `openmpi/gcc/4.1.6`, `rclone`.
- Optional: ext3 → squashfs (`mksquashfs /ext3 env.sqf -keep-as-directory`), ~2.5× smaller.

Generic recipe:

```bash
cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz $SCRATCH/env/ && gunzip $SCRATCH/env/overlay-15GB-500K.ext3.gz
apptainer exec --fakeroot \
    --overlay $SCRATCH/env/overlay-15GB-500K.ext3:rw \
    /share/apps/images/cuda-13.3.1-ubuntu-26.04.sif bash
# inside: install Miniforge into /ext3/miniforge3 (or uv into /ext3/.uv + venv /ext3/.venv), write /ext3/env.sh
```

`uv` variant: `UV_INSTALL_DIR=/ext3/.uv`, `UV_PYTHON_INSTALL_DIR=/ext3/.uv/python`, `UV_PROJECT_ENVIRONMENT=/ext3/.venv`,
then `uv sync` from the repository.

---

## 10. MULTI-NODE DDP TEMPLATE

Rules from the docs:

- tasks = GPUs: `--ntasks-per-node` = `--gres=gpu:N` (not `--ntasks`);
- fill whole nodes before going multi-node, never 1 GPU per node;
- NCCL backend, `local_rank = SLURM_PROCID % SLURM_GPUS_ON_NODE`;
- `gradient_as_bucket_view=True` saves ~1/3 of the memory;
- optimize single-GPU first (dataloader workers = `SLURM_CPUS_PER_TASK`).

Two full H200 nodes of Courant = the 16 GPUs allowed per user (`lzanna`, capped at 8 GPUs, cannot run it):

```bash
#!/bin/bash
#SBATCH --job-name=heimdall-ddp
#SBATCH --account=torch_pr_347_courant
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=16                        # 128 cores / 8 GPUs
#SBATCH --mem=2000G                               # --mem=0 is refused on Torch
#SBATCH --time=48:00:00
#SBATCH --comment="preemption=yes;requeue=true"   # optional, needs auto-resume
#SBATCH --output=cluster/logs/%x-%j.out

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun apptainer exec --nv \
    --overlay $SCRATCH/env/heimdall.ext3:ro \
    --overlay $SCRATCH/data/dataset.sqf:ro \
    /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
    bash -c "source /ext3/env.sh; RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID python train.py"
```

Job arrays share `SLURM_JOBID` ⇒ add `$SLURM_ARRAY_TASK_ID` to `MASTER_PORT`.

---

## 11. MISC

- **Acknowledgement** in publications: *"This work was supported in part through the NYU IT High Performance Computing
  resources, services, and staff expertise"*; add the papers and grants to the project in the portal.
- **Compared to Lucia** (A100 40 GB, 4 per node, 48 GPUs per user): Torch has bigger GPUs but a tighter GPU limit per
  user (16 GPUs over 2 days, 4 over 7 days), and it cancels underused GPU jobs.

---

## 12. OPEN QUESTIONS

| #   | Question                                              | Status                                     |
| --- | ----------------------------------------------------- | ------------------------------------------ |
| 1   | OSN bucket of the project? RPS volume of the group?   | `m2lines-pubs`, public (§ 8); RPS **open** |
| 2   | Delay before an underused GPU job is cancelled?       | **open**, ask hpc@nyu.edu                  |
| 3   | `courant` allocation ends on **2026-12-10**           | **to do**, PI to renew it                  |
| 4   | lzanna's GPUs and group cap                           | 8 RTX for the whole lab (§ 3)              |
| 5   | `--constraint` names, node prefix ↔ GPU type          | see § 2 and § 4                            |
| 6   | Internet on compute nodes (OSN, WandB, PyPI, GitHub)? | yes, despite the docs (§ 6)                |
| 7   | Per-user GPU cap                                      | ≤ 48 h → 16, > 48 h → 4 (§ 4)              |
