#!/bin/zsh
# Memory-pressure watchdog for tier0 overnight runs.
# Added after the 2026-07-11 02:05 kernel panic (watchdogd starved for 91s while
# mlx generation + MPS extraction ran concurrently). Kills tier0 model jobs when
# the system nears memory exhaustion, so a measurement dies instead of the Mac.
#
# Usage: nohup tier0/scripts/guard.sh >/dev/null 2>&1 &   (start before any model job)

LOGF="${0:a:h}/../logs/guard.log"
FREE_FLOOR=8   # kill below this % system-wide free memory
JOB_RE='tier0/scripts/(gen_rollouts|extract_hidden|m[0-9]_[a-z]+|t3_[a-z]+)\.py'

echo "[$(date '+%F %H:%M:%S')] guard started (floor=${FREE_FLOOR}%)" >> "$LOGF"
while true; do
  level=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null)  # 1 normal / 2 warn / 4 critical
  free=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub(/[% ]/,"",$2); print $2}')
  if [[ "$level" -ge 4 || ( -n "$free" && "$free" -lt "$FREE_FLOOR" ) ]]; then
    echo "[$(date '+%F %H:%M:%S')] pressure=$level free=${free}% -> SIGTERM tier0 jobs" >> "$LOGF"
    pkill -TERM -f "$JOB_RE"
    sleep 20
    pkill -KILL -f "$JOB_RE" && echo "[$(date '+%F %H:%M:%S')] SIGKILL survivors" >> "$LOGF"
  fi
  sleep 10
done
