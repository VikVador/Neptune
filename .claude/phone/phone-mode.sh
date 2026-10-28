#!/bin/bash
# Turns the phone answer mode on, off, or toggles it without argument, then prints the state.
state="$HOME/.claude/phone/mode"
mkdir -p "$(dirname "$state")"
case "${1:-toggle}" in
    on)     touch "$state" ;;
    off)    rm -f "$state" ;;
    toggle) if [ -f "$state" ]; then rm -f "$state"; else touch "$state"; fi ;;
    *)      echo "usage: phone-mode.sh [on|off]"; exit 1 ;;
esac
if [ -f "$state" ]; then echo "phone mode: on"; else echo "phone mode: off"; fi
