#!/usr/bin/env python3
"""Download Laguna-S-2.1 model tensors from HuggingFace.

Usage:
  python3 download_laguna.py --repo poolside/Laguna-S-2.1 --outdir ./laguna_model
  python3 download_laguna.py --repo poolside/Laguna-S-2.1 --outdir ./laguna_model --max-concurrent 2
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

def get_hf_file_list(repo):
    """Get list of files from HuggingFace API."""
    url = f"https://huggingface.co/api/models/{repo}/tree/main"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "colibri-downloader"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
        return [item["rfilename"] for item in data if item.get("rfilename", "").endswith(".safetensors")]
    except Exception as e:
        print(f"ERROR: Could not list files from HF API: {e}")
        return []

def get_hf_index(repo):
    """Get the model index file to find all shard names."""
    url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "colibri-downloader"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
        weight_map = data.get("weight_map", {})
        shards = sorted(set(weight_map.values()))
        return shards
    except Exception as e:
        print(f"  No index file found, will list files directly: {e}")
        return None

def download_file(url, output_path, max_retries=3, timeout=300):
    """Download a single file using wget."""
    cmd = [
        "wget",
        "-q",  # Quiet mode
        "--continue",  # Resume partial downloads
        f"--tries={max_retries}",
        f"--timeout={timeout}",
        "-O", output_path,
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def file_size(path):
    """Get file size in bytes."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

def format_size(size_bytes):
    """Format file size for display."""
    if size_bytes == 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def main():
    parser = argparse.ArgumentParser(description="Download Laguna-S-2.1 model from HuggingFace")
    parser.add_argument("--repo", default="poolside/Laguna-S-2.1", help="HuggingFace repo name")
    parser.add_argument("--outdir", default="./laguna_model", help="Output directory")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Max concurrent downloads")
    parser.add_argument("--max-retries", type=int, default=3, help="Max download retries")
    parser.add_argument("--timeout", type=int, default=300, help="Download timeout in seconds")
    parser.add_argument("--no-metadata", action="store_true", help="Skip downloading config/tokenizer files")
    args = parser.parse_args()

    repo = args.repo
    outdir = args.outdir
    hf_base = f"https://huggingface.co/{repo}/resolve/main"
    
    os.makedirs(outdir, exist_ok=True)
    
    print(f"Downloading from: {repo}")
    print(f"Output directory: {os.path.abspath(outdir)}")
    print(f"Max concurrent downloads: {args.max_concurrent}")
    print()

    # Get list of shard files
    print("Fetching model index...")
    shards = get_hf_index(repo)
    
    if not shards:
        print("Falling back to listing files directly...")
        shards = get_hf_file_list(repo)
    
    if not shards:
        print("ERROR: No shard files found!")
        sys.exit(1)
    
    print(f"Found {len(shards)} shard files:")
    for s in shards:
        print(f"  - {s}")
    print()

    # Download metadata files first (config, tokenizer, etc.)
    if not args.no_metadata:
        print("Downloading metadata files...")
        metadata_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "chat_template.jinja"
        ]
        for fn in metadata_files:
            url = f"{hf_base}/{fn}"
            output_path = os.path.join(outdir, fn)
            if os.path.exists(output_path) and file_size(output_path) > 0:
                print(f"  [SKIP] {fn} (already exists)")
                continue
            print(f"  [DOWNLOAD] {fn}")
            if download_file(url, output_path, args.max_retries, args.timeout):
                print(f"  [OK] {fn} ({format_size(file_size(output_path))})")
            else:
                print(f"  [FAIL] {fn}")
        print()

    # Download shard files
    print(f"Starting download of {len(shards)} shard files...")
    print(f"Max concurrent: {args.max_concurrent}")
    print()

    # Track download status
    download_queue = list(shards)
    completed = []
    failed = []
    processes = []
    
    while download_queue or any(p[0].poll() is None for p in processes):
        # Start new downloads up to max_concurrent
        while len([p for p in processes if p[0].poll() is None]) < args.max_concurrent and download_queue:
            shard = download_queue.pop(0)
            url = f"{hf_base}/{shard}"
            output_path = os.path.join(outdir, shard)
            
            if os.path.exists(output_path) and file_size(output_path) > 0:
                print(f"  [{len(completed)+len(failed)+1}/{len(shards)}] [SKIP] {shard} ({format_size(file_size(output_path))})")
                completed.append(shard)
                continue
            
            # Start download in background
            cmd = [
                "wget", "-q", "--continue",
                f"--tries={args.max_retries}",
                f"--timeout={args.timeout}",
                "-O", output_path,
                url
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append((p, shard, output_path, time.time()))
            print(f"  [{len(completed)+len(failed)+1}/{len(shards)}] [START] {shard}")
        
        # Check for completed downloads
        new_processes = []
        for p, shard, output_path, start_time in processes:
            if p.poll() is not None:
                # Download finished
                if p.returncode == 0 and file_size(output_path) > 0:
                    print(f"  [{len(completed)+len(failed)+1}/{len(shards)}] [DONE]  {shard} ({format_size(file_size(output_path))})")
                    completed.append(shard)
                else:
                    print(f"  [{len(completed)+len(failed)+1}/{len(shards)}] [FAIL]  {shard}")
                    failed.append(shard)
                    # Retry failed downloads
                    if download_queue.count(shard) == 0:
                        download_queue.append(shard)
            else:
                new_processes.append((p, shard, output_path, start_time))
        
        processes = new_processes
        
        if download_queue or processes:
            time.sleep(1)
        
        if not download_queue and not processes:
            break

    print()
    print("=" * 60)
    print("Download Summary:")
    print(f"  Total shards: {len(shards)}")
    print(f"  Completed: {len(completed)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Output directory: {os.path.abspath(outdir)}")
    print()
    
    # Show downloaded files
    print("Downloaded files:")
    for fn in sorted(os.listdir(outdir)):
        if fn.endswith(".safetensors"):
            path = os.path.join(outdir, fn)
            print(f"  {fn} - {format_size(file_size(path))}")
    
    if failed:
        print()
        print(f"WARNING: {len(failed)} downloads failed. Run the script again to retry.")
        sys.exit(1)

if __name__ == "__main__":
    main()