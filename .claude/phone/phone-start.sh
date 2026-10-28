#!/bin/bash
# At the start of the claude-phone session, forgets the last prompt time so the first prompt asks about the phone.
[ "$CLAUDE_PHONE" = "1" ] || exit 0
rm -f "$HOME/.claude/phone/last-prompt"
