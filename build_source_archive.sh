#!/usr/bin/env bash
# Build a source archive from the checked-in Git contents of this repo and
# every initialized recursive submodule. Untracked files, ignored files, build
# outputs, and local edits are deliberately excluded.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: ./build_source_archive.sh [OPTIONS]

Create a tar.gz archive containing the checked-in source for the current
repository and all initialized recursive submodules.

Options:
  --output-dir DIR   Directory for the generated archive
                     default: current repository root
  --output PATH      Exact archive path to write
  --prefix NAME      Top-level directory name inside the archive
                     default: <repo>-source-<UTC timestamp>-<short sha>
  -h, --help         Show this help

Notes:
  - Archives committed/tracked content only, using `git archive`.
  - Local uncommitted edits, untracked files, ignored files, target/, and .git/
    directories are not included.
  - Submodules must be initialized locally. Run
    `git submodule update --init --recursive` first if needed.
USAGE
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

require_value() {
    local opt="$1"
    local value="${2-}"
    if [[ -z "$value" || "$value" == --* ]]; then
        die "$opt requires a value"
    fi
}

output_dir=""
output_path=""
prefix=""

while (($#)); do
    case "$1" in
        --output-dir)
            require_value "$1" "${2-}"
            output_dir="$2"
            shift 2
            ;;
        --output)
            require_value "$1" "${2-}"
            output_path="$2"
            shift 2
            ;;
        --prefix)
            require_value "$1" "${2-}"
            prefix="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || die "not inside a Git repository"
cd "$repo_root"

git rev-parse --verify HEAD >/dev/null 2>&1 || die "repository has no HEAD commit to archive"

repo_name=$(basename "$repo_root")
short_sha=$(git rev-parse --short=12 HEAD)
timestamp=$(date -u +%Y%m%dT%H%M%SZ)

if [[ -z "$prefix" ]]; then
    prefix="${repo_name}-source-${timestamp}-${short_sha}"
fi

if [[ "$prefix" == /* || "$prefix" == *..* ]]; then
    die "prefix must be a simple relative directory name"
fi

if [[ -n "$output_path" ]]; then
    archive_path="$output_path"
    archive_dir=$(dirname "$archive_path")
else
    if [[ -n "$output_dir" ]]; then
        archive_dir="$output_dir"
        archive_path="${archive_dir}/${prefix}.tar.gz"
    else
        archive_dir="."
        archive_path="${prefix}.tar.gz"
    fi
fi

mkdir -p "$archive_dir"

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/${repo_name}-source-archive.XXXXXX")
cleanup() {
    rm -rf "$tmpdir"
}
trap cleanup EXIT

stage="$tmpdir/stage"
mkdir -p "$stage"

printf '[source-archive] repo: %s\n' "$repo_root"
printf '[source-archive] prefix: %s\n' "$prefix"
printf '[source-archive] exporting superproject HEAD %s\n' "$(git rev-parse --short=12 HEAD)"

git archive --format=tar --prefix="${prefix}/" HEAD | tar -xf - -C "$stage"

submodule_status_file="$tmpdir/submodule_status.txt"
git submodule status --recursive > "$submodule_status_file" || true

if [[ -s "$submodule_status_file" ]]; then
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        # Format: "<status><sha> <path> (<describe>)". Submodule paths in this
        # repo do not contain spaces; if that ever changes, prefer avoiding
        # spaces in submodule paths or extend this parser.
        path=$(printf '%s\n' "$line" | awk '{print $2}')
        status_char=${line:0:1}
        if [[ -z "$path" ]]; then
            die "could not parse submodule status line: $line"
        fi
        if [[ "$status_char" == "-" ]]; then
            die "submodule '$path' is not initialized; run: git submodule update --init --recursive"
        fi
        if [[ ! -d "$path" ]]; then
            die "submodule path '$path' is missing; run: git submodule update --init --recursive"
        fi
        git -C "$path" rev-parse --verify HEAD >/dev/null 2>&1 || \
            die "submodule '$path' has no checked-out HEAD"
        printf '[source-archive] exporting submodule %s HEAD %s\n' \
            "$path" "$(git -C "$path" rev-parse --short=12 HEAD)"
        mkdir -p "$stage/$prefix/$path"
        git -C "$path" archive --format=tar --prefix="${prefix}/${path}/" HEAD | tar -xf - -C "$stage"
    done < "$submodule_status_file"
fi

manifest="$stage/$prefix/SOURCE_ARCHIVE_MANIFEST.txt"
{
    printf 'Source archive manifest\n'
    printf '=======================\n\n'
    printf 'Generated UTC: %s\n' "$timestamp"
    printf 'Repository: %s\n' "$repo_name"
    printf 'Repository root: %s\n' "$repo_root"
    printf 'Archive prefix: %s\n' "$prefix"
    printf 'Superproject HEAD: %s\n' "$(git rev-parse HEAD)"
    printf 'Superproject short HEAD: %s\n' "$short_sha"
    printf '\nArchive policy:\n'
    printf '%s\n' '- Includes committed/tracked files from the superproject HEAD.'
    printf '%s\n' '- Includes committed/tracked files from each initialized recursive submodule HEAD.'
    printf '%s\n' '- Excludes local edits, untracked files, ignored files, build outputs, and .git directories.'
    printf '\nSuperproject status at archive time:\n'
    git status --short --branch || true
    printf '\nRecursive submodule status at archive time:\n'
    if [[ -s "$submodule_status_file" ]]; then
        cat "$submodule_status_file"
    else
        printf '(none)\n'
    fi
    printf '\nSubmodule HEADs included:\n'
    if [[ -s "$submodule_status_file" ]]; then
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            path=$(printf '%s\n' "$line" | awk '{print $2}')
            if [[ -n "$path" && -d "$path" ]]; then
                printf '%s %s\n' "$(git -C "$path" rev-parse HEAD)" "$path"
            fi
        done < "$submodule_status_file"
    else
        printf '(none)\n'
    fi
} > "$manifest"

tar -C "$stage" -czf "$archive_path" "$prefix"

printf '[source-archive] wrote: %s\n' "$archive_path"
printf '[source-archive] list contents: tar -tzf %q | less\n' "$archive_path"
