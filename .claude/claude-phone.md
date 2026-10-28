```text
     ┌───────┐
     │ ◉   ◉ │       C   L   A   U   D   E
     │   ▿   │
     └─┬───┬─┘       Phone: drive Claude Code from the phone while it
     ┌─┴───┴─┐       works on a cluster (Remote Control in tmux).
     │ ▓▓▓▓▓ │
     └─┘   └─┘
```

# CLAUDE PHONE

**Set up: 2026-10-26** on `torch-login-a-1` (Heimdall), **2026-10-28** on `frontal02` (Neptune, Lucia).

Drive Claude Code from the phone (Claude mobile app, or https://claude.ai/code) while it works on Lucia, through the
Remote Control of Claude Code.

**Contents**

0. In short
1. Usage
2. How it works
3. How long it stays open
4. Requirements
5. Phone answer mode
6. Install on a new machine

---

## 0. IN SHORT

1. **Start** in a VSCode terminal on Lucia: `claude-phone`. **Leave** without stopping it: **Ctrl+B**, then **D**.
2. **Phone**: Claude app → **Code** tab → session **neptune**.
3. **One active session at a time**: never give work to the phone session and to VSCode simultaneously.
4. **tmux is per node**: check or stop the session from the node where it runs; `claude-phone status` names it.
5. **Phone answer mode**: short answers for a small screen, turned on with `/phone on`, off with `/phone off` (§ 5).

---

## 1. USAGE

| Command               | Effect                                                                                         |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| `claude-phone`        | Starts the tmux session `claude` if needed, then attaches to it                                |
| `claude-phone stop`   | Kills the tmux session (and Claude with it)                                                    |
| `claude-phone status` | Tells whether the session runs **on the current node**, otherwise on which node it was started |

| Step | Where           | What                                                                            |
| ---- | --------------- | ------------------------------------------------------------------------------- |
| 1    | VSCode on Lucia | `claude-phone`                                                                  |
| 2    | same terminal   | leave without stopping Claude: **Ctrl+B**, then **D** (or close the terminal 🗑️) |
| 3    | phone           | Claude app → **Code** tab → session **neptune**                                 |

| Option (environment variable) | Effect                                         | Default       |
| ----------------------------- | ---------------------------------------------- | ------------- |
| `CLAUDE_PHONE_MODE`           | permission mode (`default`, `plan`, `auto`, …) | `acceptEdits` |
| `CLAUDE_PHONE_DIR`            | working directory                              | `~/neptune`   |

---

## 2. HOW IT WORKS

| Piece          | Details                                                                                                  |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| Script         | `~/bin/claude-phone` (in `PATH`, outside the repository)                                                 |
| Claude binary  | `claude` from `PATH` (standalone CLI in `~/.local/bin`), otherwise the latest one bundled with VSCode    |
| Command        | `claude remote-control` in a detached tmux session, from `~/neptune`                                     |
| tmux socket    | `~/.tmux/claude-phone-<node>.sock`, not `/tmp` (cleaned by the node, which made the session unreachable) |
| Login node use | `nice -n 10`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` set to 4 (shared nodes)    |
| Network        | outbound HTTPS to Anthropic: the phone needs neither the ULiège VPN nor the SSH gateway                  |

**Restart loop**

- The first run creates the session `neptune` (`--name`), the next ones reattach to it (`-c`).
- After a failure, it restarts 30 s later.
- It stops on exit code 0 or 130 (Ctrl+C); Ctrl+C during the 30 s wait also stops it.

---

## 3. HOW LONG IT STAYS OPEN

| Event                                       | Consequence                                                     |
| ------------------------------------------- | --------------------------------------------------------------- |
| Network outage of about 10 minutes          | Claude exits, the loop reattaches with `-c` every 30 s          |
| `-c` only reattaches within about **4 h**   | After 480 failed retries the loop gives up: `stop` then `start` |
| Reboot of the login node                    | Everything is lost: run `claude-phone` again                    |
| Long processes killed by the cluster admins | Same, run `claude-phone` again                                  |
| `claude-phone stop`, or exit in the session | Ends normally                                                   |

**Good to know**

- **tmux is per node.** `frontal.lucia.cenaero.be` is a DNS round robin, so VSCode can reconnect to the other login
  node (`frontal01` / `frontal02`): `claude-phone status` then says "not running" and names the node where the
  session was started. Go there with `ssh <node>` before `claude-phone stop` or attaching, otherwise you start a
  **second** session.
- **Login nodes are for light work.** The [Lucia docs](https://doc.lucia.cenaero.be/connect/ssh/) say "NO
  COMPUTATIONS are allowed on login nodes" and allow "lightweight tasks"; they say nothing about long, light
  processes such as tmux or Remote Control. There is no per-user memory cap on `frontal01` / `frontal02` (128 cores,
  251 GB shared by everyone), so the limit is ours: Claude uses ~250 MB, heavy work (training, analysis, statistics)
  goes through Slurm jobs. Ask https://support.lucia.cenaero.be in case of doubt.
- **Maintenance** (next one on Monday 30 November 2026): the login nodes reboot, run `claude-phone` again afterwards.

---

## 4. REQUIREMENTS

| Requirement     | Details                                                                       |
| --------------- | ----------------------------------------------------------------------------- |
| Claude Code CLI | ≥ 2.1, installed with `curl -fsSL https://claude.ai/install.sh \| bash`       |
| Login           | a **claude.ai account** (`/login`), API keys are not supported                |
| Remote Control  | allowed by the **personal** account, blocked by the NYU-managed one           |
| Phone           | logged in with the **same account** as the CLI                                |
| tmux            | `/usr/bin/tmux` on Torch, `/usr/bin/tmux` 2.7 on Lucia                        |
| One session     | **one active session at a time**: never the phone session and VSCode together |

The NYU-managed account shows the block in `~/.claude/policy-limits.json` (`allow_remote_control: false`).

**On Lucia (2026-10-28)**: logged in with the personal claude.ai account (Pro, `claude auth status`), no
`policy-limits.json`. No standalone CLI yet: `claude-phone` uses the binary bundled with VSCode (2.1.312). A VSCode
update removes the old binary, which breaks a running session at its next restart: the standalone CLI avoids it.

---

## 5. PHONE ANSWER MODE

In the session started by `claude-phone`, answers can switch to a **phone style**, short enough for a small screen.

| When                                        | What happens                                       |
| ------------------------------------------- | -------------------------------------------------- |
| First prompt of the session                 | Claude first asks: "Es-tu sur ton téléphone ?"     |
| More than **2 h** since the previous prompt | Same question                                      |
| Answer yes / no                             | Claude turns the mode on / off by itself           |
| `/phone on`, `/phone off`, `/phone`         | Turns it on, off, or toggles it, at any time       |
| Mode on                                     | Every prompt reminds Claude to use the phone style |

**Phone style**: every answer has three parts, separated by a horizontal rule (a `---` line between blank lines).

| Part       | Content                                                                            |
| ---------- | ---------------------------------------------------------------------------------- |
| 1. Title   | the task in a few words, in **bold**                                               |
| 2. Summary | 2 to 4 lines of plain text: what was done or found, and why, very simple and clear |
| 3. Changes | one bullet per change: *what* changed, then *why* in one short, simple sentence    |

- Part 3 is left out when nothing was changed (a simple question, a status check).
- A question to me comes last, on its own line, with the expected answers: `👉 Je relance ? (oui / non)`.
- No wide tables, no code blocks, no long paths: file names only. I can ask "détaille" for the normal answer.

Example:

```markdown
**SSS : valeurs aberrantes retirées**

---

Les pics de la validation venaient de quelques salinités impossibles (> 45 PSU)
en 2019-2021. Je les ai retirées et l'entraînement repart de zéro (≈ 20 h).

---

- `data/__init__.py` : les salinités > 45 PSU deviennent des trous (NaN),
  comme les zones sans mesure.
- `encoder-sss.yml` : relancé de zéro, car les anciens poids avaient appris ces pics.
- Job 21012345 : en attente d'une RTX du labo, démarrage estimé ce soir.

👉 Je pousse la branche ? (oui / non)
```

**How it works** (outside the repository, for all projects)

| Piece                             | Role                                                                |
| --------------------------------- | ------------------------------------------------------------------- |
| `CLAUDE_PHONE=1`                  | set by `claude-phone` in the tmux session: the hooks act only there |
| `~/.claude/hooks/phone-prompt.sh` | hook on every prompt: 2 h check, reminder when the mode is on       |
| `~/.claude/hooks/phone-start.sh`  | hook at startup: the first prompt of a new session always asks      |
| `~/.claude/hooks/phone-mode.sh`   | turns the mode on / off (state file `~/.claude/phone/mode`)         |
| `~/.claude/commands/phone.md`     | the `/phone` command                                                |
| `~/.claude/settings.json`         | declares the two hooks                                              |

---

## 6. INSTALL ON A NEW MACHINE

The scripts live in `.claude/phone/` of the repository; copy them to the places Claude Code and the shell expect.

| From `.claude/phone/`                                | To                                                         |
| ---------------------------------------------------- | ---------------------------------------------------------- |
| `claude-phone`                                       | `~/bin/claude-phone` (in `PATH`)                           |
| `phone-mode.sh`, `phone-prompt.sh`, `phone-start.sh` | `~/.claude/hooks/`                                         |
| `phone.md`                                           | `~/.claude/commands/phone.md`                              |
| `settings-hooks.json`                                | merged into `~/.claude/settings.json` (keep what is there) |

```bash
mkdir -p ~/bin ~/.claude/hooks ~/.claude/commands
cp .claude/phone/claude-phone ~/bin/ && cp .claude/phone/phone-*.sh ~/.claude/hooks/
cp .claude/phone/phone.md ~/.claude/commands/ && chmod +x ~/bin/claude-phone ~/.claude/hooks/phone-*.sh
```

- Set `CLAUDE_PHONE_DIR` (or edit the default `~/neptune` in the script) to the repository of the project.
- After a change of a script, copy it back into `.claude/phone/` so that both stay identical.
