```text
     ┌───────┐
     │ ◉   ◉ │       C   L   A   U   D   E
     │   ▿   │
     └─┬───┬─┘       Lucia cluster (Cenaero, Wallonia): access, GPUs,
     ┌─┴───┴─┐       Slurm, billing, storage and software environment.
     │ ▓▓▓▓▓ │
     └─┘   └─┘
```

# LUCIA HPC

**Last updated: 2026-10-28**

> **Keep this file up to date.** If the date above is **more than one month old**, before relying on this file:
> 1. launch an agent to re-read the [Lucia docs](https://doc.lucia.cenaero.be) and the
>    [CÉCI pages](https://www.ceci-hpc.be/clusters/lucia/), and check that nothing changed;
> 2. check your memory for anything useful learned on Lucia since then;
> 3. update this file, its date, and tell me what changed.

Distilled from the [Lucia docs](https://doc.lucia.cenaero.be) (29 pages read on 2026-10-27), completed with the
[CÉCI](https://www.ceci-hpc.be) pages (marked **(CÉCI)**) and with what Neptune already uses on Lucia.
**Checked on the cluster on 2026-10-28** (`frontal02`, `sinfo`, `sacctmgr`, `scontrol`, `mmlsquota`,
`sbatch --test-only`; no job run). **[verify]** = still missing or ambiguous.

**Contents**

0. In short
1. Access
2. Hardware
3. Our resources
4. Slurm
5. Billing
6. Storage
7. Big datasets and transfers
8. Software environment
9. GPU job template
10. Misc
11. Open questions

---

## 0. IN SHORT

1. **`--account=bsmfc` and `--partition=<name>`** in every job (unlike Torch, the partition is given).
   If `bsmfc` runs out of GPU credits, **ask me** before switching to the `sail` account.
2. **GPU partition: 4 A100 40 GB per node, at most 8 CPUs and 60 GB per GPU**, 48 GPUs per user, **48 h max**.
3. **No wall-time extension** ⇒ checkpoint/restart.
4. **Every job costs credits** (BUh): a GPU node costs 512 BUh/h; a project out of credits cannot submit.
5. **Scratch is wiped at the May and November maintenances** and has no backup.
6. **Files in projects and scratch must belong to the project group**, otherwise "Disk quota exceeded".
7. **Login through an SSH gateway** (`gwceci.uliege.be`, ULiège VPN off campus); no computation on login nodes.
8. **Apptainer only** for containers (`--nv` for GPUs), software through Lmod / EasyBuild modules.

---

## 1. ACCESS

| Item         | Details                                                                                                                                                                                     |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Requirements | a **CÉCI account** and a **project** created by a permanent member of a CÉCI university (e.g. the supervisor) at https://login.ceci-hpc.be/init-project/; projects last 3 years (CÉCI).     |
| Login        | `frontal.lucia.cenaero.be` (round robin over `frontal01` / `frontal02`), **through an SSH gateway**.                                                                                        |
| Gateway      | ULiège: `gwceci.uliege.be` with the CÉCI login and key, **ULiège VPN off campus** (CÉCI). Cenaero and industry: `gateway.lucia.cenaero.be` (no computation, 10 / 40 GB, separate password). |
| SSH config   | the CÉCI [ssh_config wizard](https://www.ceci-hpc.be/sshconfig.html), tick "Tier1". SSH keys supported (ed25519 with passphrase).                                                           |
| Password     | needed for Open OnDemand and support. Valid **365 days**; expired = no login, even with a key. 12 characters min.                                                                           |
| Web access   | Open OnDemand at https://hpcportal.lucia.cenaero.be: Jupyter, Linux desktop, ParaView, batch jobs.                                                                                          |
| Support      | https://support.lucia.cenaero.be (Redmine, HPC credentials), or it@cenaero.be.                                                                                                              |

**Good to know**

- Password: `passwd_days_left`, `passwd`, reset at https://ssp.lucia.cenaero.be. **5 wrong attempts lock the account
  for 10 min.** The first SSH login changes the initial password and creates `$HOME` (needed before using OOD).
- Login nodes: no computations, light tasks and small compilations only. Each has 128 cores and 251 GB, shared by
  every user, with **no per-user memory or CPU cap** (cgroup and `ulimit` unlimited, 2026-10-28): nothing stops a
  heavy process, so keep it light (`nice`, 4 threads). Long, light processes (tmux, Remote Control): not documented.
- A reconnection can land on the other login node (`frontal01` / `frontal02`): tmux sessions and processes started
  from the previous one keep running there (`ssh frontal01` from Lucia to reach them).
- Host-key warning: `ssh-keygen -f ~/.ssh/known_hosts -R frontal.lucia.cenaero.be`, then add the key given on the
  [SSH page](https://doc.lucia.cenaero.be/connect/ssh/).
- Support tickets: give job IDs, project, paths and error messages.
- 2FA, VSCode: not documented **[verify]**.

`~/.ssh/config` (from the docs, Cenaero gateway; academics use the CÉCI wizard instead):

```text
Host gwlucia
    Hostname gateway.lucia.cenaero.be
    User <your_username>
    IdentityFile ~/.ssh/id_rsa
    ForwardAgent yes
Host lucia
    Hostname frontal.lucia.cenaero.be
    User <your_username>
    IdentityFile ~/.ssh/id_rsa
    ProxyJump gwlucia
```

---

## 2. HARDWARE

AMD EPYC Milan (no hyperthreading), HDR InfiniBand fat tree, 10 Gb/s to Belnet. RHEL 8, Slurm 25.11.6. ~4 PFLOPS.

| Type           | Nodes (`sinfo`)      | GPUs / node            | CPUs / node | RAM / node (usable) | Local disk      |
| -------------- | -------------------- | ---------------------- | ----------: | ------------------- | --------------- |
| **GPU**        | `cna[001-050]` (50)  | **4× A100 40 GB**      |      **32** | 256 GB (**240 GB**) | none            |
| **AI**         | `cni[001-002]` (2)   | **8× A100 SXM4 80 GB** |      **64** | 2 TB (2000 GB)      | **6.4 TB NVMe** |
| Visualization  | `cng[001-004]` (4)   | 4× T4 16 GB            |          32 | 512 GB (492 GB)     | none            |
| CPU standard   | `cns[001-270]` (270) | –                      |         128 | 256 GB (240 GB)     | none            |
| CPU medium     | `cnm[001-030]` (30)  | –                      |         128 | 512 GB (492 GB)     | none            |
| Large / XLarge | `cnl`, `cnx` (7 + 1) | –                      |          64 | 2 TB / 4 TB         | none            |

**Good to know**

- 200 A100 40 GB in total: `cna[001-048]` in `gpu`, `cna[049-050]` in `debug-gpu`. Slurm GRES: `gpu:A100:4`
  (`cna`), `gpu:A100:8` (`cni`), `gpu:T4:4` (`cng`). NVLink between the GPUs of a node: not documented **[verify]**.
- CPU nodes: `cns[002-255]` in `batch`, `cns[261-270]` in `debug`, `cns[256-260]` and `cns001` in the private
  partitions `sab` and `crm` (not reachable by our accounts); `cnm[001-020]` in `medium`, `cnm[021-030]` in `shared`.
- Compared to Torch: **more GPUs per user (48)**, but older and smaller GPUs (A100 40 GB vs H200 150 GB).

---

## 3. OUR RESOURCES

From the Neptune repository, from Victor (2026-10-27) and from the cluster (2026-10-28).

| What            | Value                                                                                                    |
| --------------- | -------------------------------------------------------------------------------------------------------- |
| Slurm accounts  | **`bsmfc`** (MAST group, Marilaure Grégoire): **use it first**; `sail` (SAIL lab, Gilles Louppe): backup |
| Associations    | both accounts reach the same partitions: `gpu`, `ia`, `debug-gpu`, `visu`, `debug`, `shared`, `large`,   |
|                 | `xlarge`, plus `batch` / `medium` with the QoS `normal` (default) or `long`                              |
| Partitions used | `gpu` (trainings), `ia` (fine-tuning), `debug-gpu`, `visu`, `large` (statistics)                         |
| Project space   | `/gpfs/projects/acad/bsmfc/` (NEMO simulation in `nemo4.2.0/`, datasets)                                 |
|                 | `sail`: `/gpfs/projects/acad/sail`                                                                       |
| Scratch         | `/gpfs/scratch/acad/bsmfc/vmangele/` (group `bsmfc`); `sail`: `/gpfs/scratch/acad/sail`                  |
| Job submission  | Python with [dawgz](https://github.com/francois-rozet/dawgz) (`schedule(...)`), version 1.0.4            |
| Environment     | micromamba environment `neptune` (`~/micromamba/envs/neptune`): PyTorch 2.5.1, ruff 0.4.8, wandb 0.26.1  |
| Credits         | `get_slurm_quota` (both accounts)                                                                        |

Credits on 2026-10-28 (`get_slurm_quota`):

| Account |   Used BUh | Credits BUh | Left           | Expires        |
| ------- | ---------: | ----------: | -------------- | -------------- |
| `bsmfc` | 31 443 476 |  40 000 000 | 8.6 M (21 %)   | 2027-09-21     |
| `sail`  | 36 640 950 |  40 000 000 | 3.4 M (8 %)    | **2027-02-22** |

**Good to know**

- **`bsmfc` first.** When its GPU credits run low (`get_slurm_quota`), **ask me** whether to switch to `sail`:
  never switch on your own.
- 8.6 M BUh = about 16 800 hours of full GPU nodes (512 BUh/h), e.g. 175 trainings of 48 h on one node.

---

## 4. SLURM

### Rules

| Rule                             | Details                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------ |
| `--account` and `--partition`    | in every job, as in all the examples of the docs.                              |
| **≤ 8 CPUs and ≤ 60 GB per GPU** | on the `gpu` partition; a full node = `--gpus=4 --cpus-per-task=8 --mem=240G`. |
| GPU syntax                       | `--gpus=N` (docs) or `--gres=gpu:N`: both accepted (`--test-only`).            |
| **No wall-time extension**       | 48 h max on the GPU partitions ⇒ checkpoint/restart.                           |

### Partitions and limits

Live values: `sacctmgr show qos format=Name,MaxTRESPU%16,MaxJobsPU,MaxSubmitPU,MaxTRES%32,MaxWall`.

| Partition         | Nodes               | Max wall    | Limits                                            | Use                     |
| ----------------- | ------------------- | ----------- | ------------------------------------------------- | ----------------------- |
| **`gpu`**         | 48 × 4 A100 40 GB   | **48 h**    | **48 GPUs, 12 nodes per user**                    | trainings               |
| **`ia`**          | 2 × 8 A100 80 GB    | **48 h**    | –                                                 | big models, fine-tuning |
| **`debug-gpu`**   | 2 × 4 A100 40 GB    | **2 h**     | 2 nodes per job, 1 running and 10 queued per user | tests                   |
| `visu`            | 4 × 4 T4            | 4 h         | exactly 1 GPU, ≤ 8 CPUs, ≤ 123 GB, 1 job          | light GPU work          |
| `batch`, `medium` | 254 / 20 × 128 CPUs | 48 h        | whole nodes (ExclusiveUser), 2000 queued jobs     | CPU jobs                |
| `batch` (`long`)  | 128 CPUs            | 168 h       | 4 nodes, 512 CPUs per user                        | long CPU jobs           |
| `shared`          | 10 × 128 CPUs       | 168 h       | 1 node per job, 500 queued                        | small CPU jobs          |
| `debug`           | 10 × 128 CPUs       | 2 h         | 4 nodes, 4 running, 20 queued                     | CPU tests               |
| `large`, `xlarge` | 7 × 2 TB / 1 × 4 TB | 168 / 336 h | ≥ 490 GB / ≥ 1000 GB per job; 4 / 1 nodes per user | big memory             |

**Good to know**

- **No preemption**: `PreemptMode=OFF` on `gpu`, `ia`, `debug-gpu` and `batch` (2026-10-28). Cancellation of
  underused GPUs and requeue: not documented, nothing seen in the configuration **[verify]**.
- `long` QoS: `--partition=batch --qos=long` (accepted by `--test-only`). The partitions do not overlap.
- **GPU nodes are shared** between jobs (`ExclusiveUser=NO`, `OverSubscribe=NO`): a 1-GPU job gets 8 CPUs of a node
  (`--test-only`). Default memory: 7.5 GB per CPU on `gpu` / `debug-gpu` (= 60 GB per GPU), 31 GB on `ia`.
- `xlarge`: the partition says 7 days, but its QoS overrides it with 14 days (`PartitionTimeLimit`).
- Interactive: `srun -p debug-gpu -A bsmfc --gpus=1 -t 60 --pty bash`, or `salloc -p batch -A bsmfc -N 1 -t 1:00:00`.
- Arrays: `#SBATCH --array=0-19`, output `%A-%a_%x.out`, `$SLURM_ARRAY_TASK_ID`.

---

## 5. BILLING

Counted in **BUh**: the largest of requested CPUs, memory and GPUs (in CPU equivalents) × weight of the node × hours.

| Partition     | Weight |       1 GPU |        Full node |
| ------------- | -----: | ----------: | ---------------: |
| `gpu`         |     16 |   128 BUh/h |    **512 BUh/h** |
| `ia`          |   25.6 | 204.8 BUh/h | **1638.4 BUh/h** |
| `visu`        |      4 |    32 BUh/h |        128 BUh/h |
| `batch` (CPU) |      1 |           – |        128 BUh/h |

**Good to know**

- Billing weights (`scontrol show partition`): `gpu` = CPU 16, memory 2.13 per GB, GPU 128, so the largest of the
  three gives 512 BUh/h for a full node; `ia` = CPU 25.6, memory 0.82 per GB, GPU 204.8.
- A project out of credits, or expired, **cannot submit**. Top-up: https://login.ceci-hpc.be/init-project/#update_project.
- Usage: `get_slurm_quota`, or
  `sreport cluster AccountUtilizationByUser Accounts=bsmfc -t Hour -T billing Start=... End=...`.
- Fair share: 85 % for non-economic projects (70 % universities), shared equally between the projects of a category.

---

## 6. STORAGE

IBM Spectrum Scale (GPFS), about 3 PiB.

| Path                                | Default quota (space / files) | Backup                          | Purge                                          | Use                          |
| ----------------------------------- | ----------------------------- | ------------------------------- | ---------------------------------------------- | ---------------------------- |
| `/gpfs/home/<org>/<user>` (`$HOME`) | 200 GB / ~1 M                 | snapshots 7 days + tape 30 days | no                                             | code, configs                |
| `/gpfs/projects/<org>/<project>`    | 1000 GB / ~1 M                | snapshots 7 days + tape 30 days | no                                             | shared data, long term       |
| `/gpfs/scratch/<org>/<project>`     | 1000 GB / ~1 M                | **no**                          | **wiped at the May and November maintenances** | fast I/O (NVMe burst buffer) |

Our quotas on 2026-10-28 (`mmlsquota`, soft quota / hard limit):

| Space                      | Used        | Quota / limit (space) | Files used | Quota (files) |
| -------------------------- | ----------- | --------------------- | ---------- | ------------- |
| home `vmangele`            | 109 GB      | 200 / 230 GB          | **940 K**  | **1.02 M** ⚠  |
| projects `bsmfc` (group)   | 99.6 TB     | 150 / 195 TB          | 1.9 M      | 10.2 M        |
| scratch `bsmfc` (group)    | **121.5 TB**| 150 / 195 TB          | 2.1 M      | 10.2 M        |

**Good to know**

- **Home: 92 % of the files used** (environments, `.vscode-server`, caches): avoid new environments or many small
  files in `$HOME`. The default quotas of the docs (and CÉCI's 50 / 50 / 100 GB) do not apply to `bsmfc`.
- Check: `mmlsquota -g <project> --block-size g ess:projects` (or `ess:scratch`, `ess:home` with `-u <user>`).
- **Group trap**: files in projects and scratch must belong to the **project group**, otherwise "Disk quota exceeded".
  Use `newgrp <project>`, copy with `rsync -av --no-p --no-g --chmod=ug=rwX` or `cp` (not `mv`, not `scp`).
- Snapshots: `~/.snapshots/` (restore with `cp -p`); tape restores through a ticket.
- Scratch purge: announced 15 days before each maintenance.

---

## 7. BIG DATASETS AND TRANSFERS

- I/O-heavy work goes to **scratch** (burst buffer, ~15× faster than projects); copy results elsewhere afterwards.
- Avoid millions of small files (file quotas). Only the AI nodes have a local NVMe disk (6.4 TB).
- Transfers: `rsync` / `scp` through the SSH gateway. No DTN, no Globus documented **[verify]**.
- Internet: yes from the login nodes (GitHub, PyPI, `api.wandb.ai`, `claude.ai` reached on 2026-10-28;
  `apptainer pull docker://...` works there); from compute nodes **[verify]**.

---

## 8. SOFTWARE ENVIRONMENT

| Tool       | Details                                                                                                        |
| ---------- | -------------------------------------------------------------------------------------------------------------- |
| Modules    | Lmod: `module avail`, `module spider`, `ml`; sticky modules need `module --force purge`.                       |
| Stacks     | `EasyBuild/2022a`, `2023a`, `2024a` (default): load one first, never mix them; updated each February.          |
| CUDA       | `CUDA/11.7.0` (2022a), `CUDA/12.2.0` (2023a), `CUDA/12.6.0` (2024a); A100 = `sm_80`.                           |
| Python     | `devel/python/Anaconda3-2022.05`, `Python/3.11` and `3.12` + `virtualenv` (2023a).                             |
| Containers | **Apptainer only** (`ml containers/apptainer`), GPUs with `--nv`.                                              |
| MPI        | OpenMPI + UCX, CUDA-aware, e.g. `OpenMPI/5.0.3-NVHPC-25.1-CUDA-12.6.0` (2024a); not Intel MPI.                 |
| Own builds | EasyBuild in user space: `module load EasyBuild/2023a EasyBuild-user/4.9.0`, installs in `~/.local/easybuild`. |

**Good to know**

- Neptune uses the micromamba environment `neptune` (§ 3): activate it before running anything.
- The [cybersecurity page](https://doc.lucia.cenaero.be/cyber/) says: "Do not install software from unverified
  sources" and "Use only the modules and environments available via module managers". Install only official
  releases (checksum verified) and ask me first.
- No documented policy for conda, pip or PyTorch **[verify]**; the only PyTorch example is a container:
  `apptainer pull pytorch.sif docker://pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime`.
- Set `APPTAINER_CACHEDIR` to the project scratch (the default `~/.apptainer` grows large).
- UCX tuning: `UCX_NET_DEVICES=mlx5_0:1`, `UCX_TLS=rc,self,sm`, `UCX_MEMTYPE_CACHE=n`.

---

## 9. GPU JOB TEMPLATE

One full GPU node (4 A100), as in the docs, adapted to PyTorch DDP. **Not tested on Lucia yet** (the docs give no
PyTorch or DDP example) **[verify]**.

```bash
#!/bin/bash
#SBATCH --job-name=neptune-train
#SBATCH --account=bsmfc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8              # at most 8 CPUs per GPU
#SBATCH --mem=240G                     # at most 60 GB per GPU
#SBATCH --time=48:00:00
#SBATCH --output=%j_%x.out

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$SLURM_NTASKS

srun bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID python train.py"
```

---

## 10. MISC

- **Acknowledgement** (required): *"The present research benefited from computational resources made available on
  Lucia, the Tier-1 supercomputer of the Walloon Region, infrastructure funded by the Walloon Region under the grant
  agreement n°1910247."*
- **Maintenance**: quarterly, Mondays 07:00–19:00. Next one **Monday 30 November 2026** (scratch wiped; confirmed by
  the banner of the docs on 2026-10-28), then planned
  2027-02-22, 2027-05-10, 2027-09-13, 2027-11-29 **[verify]**.
- Lucia is mainly meant for large jobs; short wall times favour the rotation of the jobs.
- Security: never share credentials or keys, report a lost key to support immediately; Lucia is not a backup system.

---

## 11. OPEN QUESTIONS

| #   | Question                                                 | Status                                           |
| --- | -------------------------------------------------------- | ------------------------------------------------ |
| 1   | Internet from compute nodes (WandB, PyPI, GitHub)?       | **open**, test from a job                        |
| 2   | Policy for conda / pip, recommended PyTorch install?     | **open**, ask support (see § 8, cybersecurity)   |
| 3   | Cancellation of underused GPUs, preemption?              | no preemption (§ 4); cancellation **open**       |
| 4   | `--gpus` or `--gres=gpu:N`; are GPU nodes shared?        | both work, nodes shared (§ 4)                    |
| 5   | Quotas of the `bsmfc` project (home, projects, scratch)? | 150 TB per group space, home 92 % of files (§ 6) |
| 6   | Cores of the AI nodes (32 or 64)?                        | 64 (§ 2)                                         |
| 7   | Long, light processes on login nodes (tmux, Claude)?     | **open**, not documented, ask support            |
