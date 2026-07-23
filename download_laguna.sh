#!/bin/bash
# Download Laguna-S-2.1 model tensors from HuggingFace using wget
# Usage: ./download_laguna.sh [repo] [--outdir DIR] [max_concurrent]
# Example: ./download_laguna.sh poolside/Laguna-S-2.1 --outdir /Volumes/files/laguna 2
# Example: ./download_laguna.sh poolside/Laguna-S-2.1 /Volumes/files/laguna 2

set -e

REPO=""
OUTDIR=""
MAX_CONCURRENT=""

# Parse arguments: support both positional and --outdir flag
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir)
            OUTDIR="$2"
            shift 2
            ;;
        --outdir=*)
            OUTDIR="${1#*=}"
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

REPO="${POSITIONAL[0]:-poolside/Laguna-S-2.1}"
OUTDIR="${OUTDIR:-${POSITIONAL[1]:-./laguna_model}}"
MAX_CONCURRENT="${POSITIONAL[2]:-2}"
HF_BASE="https://huggingface.co/${REPO}/resolve/main"

# Resolve to absolute path
OUTDIR="$(cd "$(dirname "$OUTDIR")" 2>/dev/null && echo "$(pwd)/$(basename "$OUTDIR")" || echo "$OUTDIR")"

mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo "Downloading from: $REPO"
echo "Output directory: $(pwd)"
echo "Max concurrent downloads: $MAX_CONCURRENT"
echo ""

# First, download the model index to get the list of files
INDEX_FILE="${REPO//\//_}_index.json"
INDEX_URL="https://huggingface.co/${REPO}/resolve/main/model.safetensors.index.json"

echo "Fetching model index..."
wget -q -O "$INDEX_FILE" "$INDEX_URL" || {
    echo "ERROR: Could not download model index from $INDEX_URL"
    echo "Trying to list files directly..."
    # If no index, try to list files from the HF API
    API_URL="https://huggingface.co/api/models/${REPO}/tree/main"
    wget -q -O "files.json" "$API_URL" || {
        echo "ERROR: Could not list files from HF API either"
        exit 1
    }
    # Extract safetensors file names
    FILES=$(python3 -c "
import json, sys
with open('files.json') as f:
    data = json.load(f)
for item in data:
    if item.get('rfilename', '').endswith('.safetensors'):
        print(item['rfilename'])
" 2>/dev/null || jq -r '.[].rfilename' files.json 2>/dev/null | grep '\.safetensors$')
    rm -f files.json
}

# Extract shard file names from the index
if [ -f "$INDEX_FILE" ]; then
    echo "Extracting shard file names from index..."
    SHARDS=$(python3 -c "
import json
with open('$INDEX_FILE') as f:
    data = json.load(f)
weight_map = data.get('weight_map', {})
shards = set(weight_map.values())
for s in sorted(shards):
    print(s)
" 2>/dev/null || jq -r '.weight_map | to_entries | map(.value) | unique | .[]' "$INDEX_FILE" 2>/dev/null)
    rm -f "$INDEX_FILE"
else
    SHARDS="$FILES"
fi

if [ -z "$SHARDS" ]; then
    echo "ERROR: No shard files found"
    exit 1
fi

# Download config and tokenizer files first
echo "Downloading config and tokenizer files..."
for fn in config.json tokenizer.json tokenizer_config.json generation_config.json; do
    URL="${HF_BASE}/${fn}"
    if wget -q --spider "$URL" 2>/dev/null; then
        wget -q "$URL" -O "$fn"
        echo "  Downloaded: $fn"
    else
        echo "  Not found: $fn (skipping)"
    fi
done

echo ""
echo "Starting download of model shards..."
echo "Shards to download:"
echo "$SHARDS" | sed 's/^/  /'
echo ""

# Download shards with concurrency control
download_shard() {
    local shard="$1"
    local url="${HF_BASE}/${shard}"
    local outfile="$shard"
    
    # Check if already downloaded
    if [ -f "$outfile" ]; then
        local filesize=$(stat -f%z "$outfile" 2>/dev/null || stat -c%s "$outfile" 2>/dev/null || echo 0)
        if [ "$filesize" -gt 0 ]; then
            echo "  [SKIP] $shard (already downloaded, ${filesize} bytes)"
            return 0
        fi
    fi
    
    echo "  [START] $shard"
    if wget -q --continue --tries=3 --timeout=300 "$url" -O "$outfile"; then
        local filesize=$(stat -f%z "$outfile" 2>/dev/null || stat -c%s "$outfile" 2>/dev/null || echo 0)
        echo "  [DONE]  $shard (${filesize} bytes)"
    else
        echo "  [FAIL]  $shard (will retry)"
        return 1
    fi
}

# Process shards with concurrency
PIDS=()
COUNT=0
TOTAL=$(echo "$SHARDS" | wc -l | tr -d ' ')
COMPLETED=0
FAILED=0

for shard in $SHARDS; do
    # Wait if we've reached max concurrent downloads
    while [ ${#PIDS[@]} -ge $MAX_CONCURRENT ]; do
        NEW_PIDS=()
        for pid in "${PIDS[@]}"; do
            [ -z "$pid" ] && continue
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid"
                RESULT=$?
                if [ $RESULT -eq 0 ]; then
                    COMPLETED=$((COMPLETED + 1))
                else
                    FAILED=$((FAILED + 1))
                fi
            else
                NEW_PIDS+=("$pid")
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
        sleep 1
    done
    
    download_shard "$shard" &
    PIDS+=($!)
    COUNT=$((COUNT + 1))
    echo "  Progress: $COUNT/$TOTAL started, $COMPLETED completed, $FAILED failed"
done

# Wait for remaining downloads
echo ""
echo "Waiting for remaining downloads..."
for pid in "${PIDS[@]}"; do
    [ -z "$pid" ] && continue
    wait "$pid"
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
    echo "  Progress: $COMPLETED completed, $FAILED failed"
done

echo ""
echo "Download complete!"
echo "  Total shards: $TOTAL"
echo "  Completed: $COMPLETED"
echo "  Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Some downloads failed. Retrying failed shards..."
    for shard in $SHARDS; do
        if [ ! -f "$shard" ] || [ $(stat -f%z "$shard" 2>/dev/null || stat -c%s "$shard" 2>/dev/null || echo 0) -eq 0 ]; then
            echo "  Retrying: $shard"
            wget -q --tries=5 --timeout=300 "${HF_BASE}/${shard}" -O "$shard" || echo "  Failed: $shard"
        fi
    done
fi

echo ""
echo "Files in $(pwd):"
ls -lh *.safetensors 2>/dev/null || echo "No safetensors files found"
