#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_PY="$ROOT_DIR/.venv/bin/python"
LOG_DIR="$ROOT_DIR/.run_logs"
PID_DIR="$ROOT_DIR/.run_pids"
ENV_FILE="$ROOT_DIR/.env"

mkdir -p "$LOG_DIR" "$PID_DIR"

WEBHOOK_PID_FILE="$PID_DIR/webhook.pid"
NGROK_PID_FILE="$PID_DIR/ngrok.pid"
SERVER_PID_FILE="$PID_DIR/server.pid"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1"
    exit 1
  fi
}

is_running() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
}

stop_pid_file() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && is_running "$pid"; then
      kill "$pid" 2>/dev/null || true
      sleep 0.5
      if is_running "$pid"; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
    rm -f "$pid_file"
  fi
}

update_env_webhook_url() {
  local new_url="$1"
  local key="WEBHOOK_UPLOAD_URL"
  if grep -q "^${key}=" "$ENV_FILE"; then
    sed -i '' "s|^${key}=.*|${key}=${new_url}|" "$ENV_FILE"
  else
    printf "\n%s=%s\n" "$key" "$new_url" >> "$ENV_FILE"
  fi
}

load_env() {
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
}

wait_for_ngrok_url() {
  local tries=30
  local delay=1
  local url=""
  for ((i=1; i<=tries; i++)); do
    url="$(curl -s http://127.0.0.1:4040/api/tunnels | "$VENV_PY" -c 'import json,sys; d=json.load(sys.stdin); ts=d.get("tunnels",[]); print(next((t.get("public_url","") for t in ts if str(t.get("public_url","")).startswith("https://")), ""))' 2>/dev/null || true)"
    if [[ "$url" == https://* ]]; then
      echo "$url"
      return 0
    fi
    sleep "$delay"
  done
  return 1
}

start_stack() {
  require_file "$VENV_PY"
  require_file "$ENV_FILE"
  require_file "$ROOT_DIR/webhook_server.py"
  require_file "$ROOT_DIR/server.py"
  require_file "$ROOT_DIR/setup_agent_tool.py"

  echo "Stopping old processes..."
  stop_pid_file "$WEBHOOK_PID_FILE"
  stop_pid_file "$NGROK_PID_FILE"
  stop_pid_file "$SERVER_PID_FILE"
  pkill -f "python webhook_server.py" 2>/dev/null || true
  pkill -f "python server.py" 2>/dev/null || true
  pkill -f "ngrok http 8081" 2>/dev/null || true

  echo "Starting webhook server on :8081..."
  load_env
  nohup "$VENV_PY" "$ROOT_DIR/webhook_server.py" > "$LOG_DIR/webhook.log" 2>&1 &
  echo $! > "$WEBHOOK_PID_FILE"

  echo "Starting ngrok on :8081..."
  nohup ngrok http 8081 > "$LOG_DIR/ngrok.log" 2>&1 &
  echo $! > "$NGROK_PID_FILE"

  echo "Waiting for ngrok public URL..."
  local ngrok_url
  if ! ngrok_url="$(wait_for_ngrok_url)"; then
    echo "Failed to read ngrok URL from http://127.0.0.1:4040/api/tunnels"
    echo "Check $LOG_DIR/ngrok.log"
    exit 1
  fi

  local upload_url="${ngrok_url}/upload-frame"
  local tool_url="${ngrok_url}/tools/describe_scene"

  echo "Updating .env WEBHOOK_UPLOAD_URL..."
  update_env_webhook_url "$upload_url"
  load_env

  echo "Registering ElevenLabs tool..."
  "$VENV_PY" "$ROOT_DIR/setup_agent_tool.py" --webhook-url "$tool_url" > "$LOG_DIR/register_tool.log" 2>&1 || true

  echo "Starting app server on :8080 (engine: amd)..."
  nohup "$VENV_PY" "$ROOT_DIR/server.py" --engine amd --voice elevenlabs > "$LOG_DIR/server.log" 2>&1 &
  echo $! > "$SERVER_PID_FILE"

  echo
  echo "Stack started."
  echo "ngrok URL: $ngrok_url"
  echo "Dashboard: http://localhost:8080"
  echo "Health:    ${ngrok_url}/health"
  echo
  echo "Logs:"
  echo "  tail -f $LOG_DIR/webhook.log"
  echo "  tail -f $LOG_DIR/server.log"
  echo "  tail -f $LOG_DIR/ngrok.log"
}

stop_stack() {
  echo "Stopping stack..."
  stop_pid_file "$SERVER_PID_FILE"
  stop_pid_file "$WEBHOOK_PID_FILE"
  stop_pid_file "$NGROK_PID_FILE"
  pkill -f "python server.py" 2>/dev/null || true
  pkill -f "python webhook_server.py" 2>/dev/null || true
  pkill -f "ngrok http 8081" 2>/dev/null || true
  echo "Stopped."
}

status_stack() {
  for name in webhook ngrok server; do
    pid_file="$PID_DIR/${name}.pid"
    if [[ -f "$pid_file" ]]; then
      pid="$(cat "$pid_file" 2>/dev/null || true)"
      if [[ -n "$pid" ]] && is_running "$pid"; then
        echo "$name: running (pid $pid)"
      else
        echo "$name: not running (stale pid file)"
      fi
    else
      echo "$name: not running"
    fi
  done
}

logs_stack() {
  touch "$LOG_DIR/webhook.log" "$LOG_DIR/server.log" "$LOG_DIR/ngrok.log"
  tail -n 40 -f "$LOG_DIR/webhook.log" "$LOG_DIR/server.log" "$LOG_DIR/ngrok.log"
}

case "${1:-start}" in
  start) start_stack ;;
  stop) stop_stack ;;
  restart) stop_stack; start_stack ;;
  status) status_stack ;;
  logs) logs_stack ;;
  *)
    echo "Usage: $0 [start|stop|restart|status|logs]"
    exit 1
    ;;
esac
