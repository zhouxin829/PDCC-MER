#!/usr/bin/env bash

resolve_project_dir() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"

  if [[ -n "${PROJECT_DIR:-}" ]]; then
    cd "$PROJECT_DIR" && pwd
    return
  fi

  if [[ -f "$script_dir/../../pdcc_main.py" ]]; then
    cd "$script_dir/../.." && pwd
    return
  fi

  if [[ -f "$script_dir/../pdcc_main.py" ]]; then
    cd "$script_dir/.." && pwd
    return
  fi

  pwd
}

require_project_file() {
  local project_dir="$1"
  local file_name="$2"

  if [[ ! -f "$project_dir/$file_name" ]]; then
    echo "Required file not found: $project_dir/$file_name" >&2
    echo "Set PROJECT_DIR=/path/to/pdcc or run the script from the repository root." >&2
    exit 2
  fi
}

csv_to_array() {
  local csv="$1"
  local out_name="$2"
  IFS=',' read -r -a "$out_name" <<< "$csv"
}
