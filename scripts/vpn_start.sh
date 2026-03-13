#!/usr/bin/env bash
set -euo pipefail

CONFIG=""
PID_FILE=""
LOG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --pid-file)
      PID_FILE="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CONFIG" || -z "$PID_FILE" || -z "$LOG_FILE" ]]; then
  echo "usage: vpn_start.sh --config <ovpn> --pid-file <pid> --log-file <log>" >&2
  exit 2
fi

sudo openvpn \
  --config "$CONFIG" \
  --daemon \
  --writepid "$PID_FILE" \
  --log "$LOG_FILE"
