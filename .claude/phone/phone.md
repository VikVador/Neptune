---
description: Turn the phone answer mode on or off (toggles without argument)
argument-hint: on | off
allowed-tools: Bash(~/.claude/hooks/phone-mode.sh:*)
---

Run `~/.claude/hooks/phone-mode.sh $ARGUMENTS` with Bash, then confirm the new state in one short line in French.
When the mode is on, answer in the phone style of `claude-phone.md` (section 5) from now on; when it is off, answer
normally. The mode only acts in the session started by `claude-phone`.
