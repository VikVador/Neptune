#!/bin/bash
# In the claude-phone session, asks about the phone after a 2 h break and recalls the phone style when it is on.
[ "$CLAUDE_PHONE" = "1" ] || exit 0
dir="$HOME/.claude/phone"
mkdir -p "$dir"
now=$(date +%s)
previous=$(cat "$dir/last-prompt" 2>/dev/null || echo 0)
echo "$now" > "$dir/last-prompt"
if [ $((now - previous)) -gt 7200 ]; then
    echo "PHONE CHECK (claude-phone session, first prompt or more than 2 h since the last one): before anything else, ask the user in one short line whether they are on their phone. Yes: run ~/.claude/hooks/phone-mode.sh on and answer in the phone style (claude-phone.md, section 5). No: run ~/.claude/hooks/phone-mode.sh off and answer normally."
elif [ -f "$dir/mode" ]; then
    echo "PHONE MODE ON: answer in the phone style of claude-phone.md, section 5: bold concise title, ---, a 2-4 line plain summary, ---, one bullet per change (what changed, then why in one simple sentence); no wide tables or code blocks."
fi
