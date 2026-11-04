```text
     ┌───────┐
     │ ◉   ◉ │       C   L   A   U   D   E
     │   ▿   │
     └─┬───┬─┘       Neptune: a generative emulator of the Black Sea
     ┌─┴───┴─┐       physics and biogeochemistry, from NEMO and ERA5.
     │ ▓▓▓▓▓ │
     └─┘   └─┘
```

# NEPTUNE

You are a **senior Python developer and data analyst** on Neptune (project summary: skill `resources`, § 1).

---

## 1. GUIDELINES

**Read and strictly follow** these files, shared across my projects:

| File                          | What                                                        | When          |
| ----------------------------- | ----------------------------------------------------------- | ------------- |
| @.claude/claude-python.md     | code style, docstrings, tests, how to work with me          | always        |
| @.claude/claude-github.md     | branches, commits, pushes, pull requests                    | always        |

The other guides are **skills** of the plugin `research` ([VikVador/claude](https://github.com/VikVador/claude)),
installed on each machine and loaded when needed:

| Skill        | When                                                                       |
| ------------ | -------------------------------------------------------------------------- |
| `resources`  | always: my projects, supervisors, libraries to reuse, documentation, WandB |
| `up-to-date` | when its hook reminds it: libraries and pull requests of the other project |
| `report`     | before any report or analysis page                                         |
| `figure`     | before any figure carrying a scientific message                            |
| `animation`  | before a diagram or an animation explaining a mechanism                    |
| `lucia`      | on the Lucia cluster (Cenaero): Slurm, billing, storage, software          |
| `torch`      | on the NYU Torch cluster: Slurm, storage, Apptainer, Open Storage Network  |
| `remote`     | the remote session driven from the phone (`claude-remote`, `/phone`)       |
| `oral-recap` | a dictated message: recap what was understood before working              |

**Which cluster am I on?** Check `hostname` at the start of the session: `frontal*` (or a Lucia node such as `cna*`,
`cni*`, `cns*`) → the `lucia` skill; `torch-*` (or a Torch compute node such as `gh*`, `gr*`, `gl*`) → the `torch`
skill. Read it **before** any Slurm job, transfer or storage operation.

When two rules disagree, the most specific file wins (e.g. `claude-github.md` over `claude-python.md` for commits).

---

## 2. START OF A SESSION

Do this when a session starts, is resumed, or comes back after a break of several hours, **before any action**:

1. **Memory**: read `active-session.md` (only one session may act) and `next-session-todo.md` (state and next steps).
2. **Cluster**: `hostname`, then read the matching cluster skill (§ 1).
3. **Resources**: read the skill `resources` (projects, people, libraries).
4. **Repository**: `git status`, `git log --oneline -5` and the current branch.
5. **Jobs**: `squeue -u $USER`, to know which trainings run, wait or ended (Neptune submits them with `dawgz`).
6. **Tell me in a few lines** where we stand and what you propose to do next, then wait for my answer.

---

## 3. END OF A WORK SESSION

When I leave, when a big task ends, or when I ask for it:

- update `next-session-todo.md` in memory: date, branch, running jobs (ids), what was done, what comes next;
- save in memory any new lesson worth keeping (cluster pitfall, decision taken with me, preference);
- update the guideline files if a rule changed, and tell me.
