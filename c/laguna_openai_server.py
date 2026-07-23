#!/usr/bin/env python3
"""OpenAI-compatible API server for the Laguna S 2.1 model.

Provides OpenAI-compatible endpoints with streaming support:
  - GET  /v1/models
  - GET  /health
  - GET  /profile
  - GET  /experts
  - POST /v1/chat/completions (with streaming)
  - POST /v1/completions (with streaming)

Usage:
  python3 laguna_openai_server.py --model ~/Documents/Personal_projects/laguna_i4 --port 8000 [--warm-cache] [--metal]
"""
import argparse
import json
import os
import sys
import threading
import time
import subprocess
import re
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

class LagunaEngine:
    """Manages laguna engine subprocess calls."""
    def __init__(self, model_dir, engine_path, cap=64, max_tokens=1024, ram_gb=20, warm_cache=False, use_metal=False, ctx=1024, kv_i8=True, persistent=False):
        self.model_dir = model_dir
        self.engine_path = engine_path
        self.cap = cap
        self.max_tokens = max_tokens
        self.ram_gb = ram_gb
        self.lock = threading.Lock()
        self.tokenizer = None
        self.warm_cache = warm_cache
        self.use_metal = use_metal
        self.ctx = ctx
        self.kv_i8 = kv_i8
        self.persistent = persistent
        self.process = None
        self._load_tokenizer()
        
        if warm_cache:
            sys.stderr.write("[warmup] Pre-loading model files into OS file cache...\n")
            sys.stderr.flush()
            self._warm_file_cache()
        
        if persistent:
            self._start_engine()
    
    def _load_tokenizer(self):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(os.path.join(self.model_dir, "tokenizer.json"))
    
    def _warm_file_cache(self):
        """Pre-read model files to warm the OS file cache."""
        import glob
        shards = sorted(glob.glob(os.path.join(self.model_dir, "out-*.safetensors")))
        total_size = 0
        for shard in shards:
            with open(shard, 'rb') as f:
                while True:
                    chunk = f.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    total_size += len(chunk)
        sys.stderr.write(f"[warmup] Pre-loaded {total_size / 1e9:.1f} GB into file cache\n")
        sys.stderr.flush()
    
    def _start_engine(self):
        """Start the engine in serve mode for persistent operation."""
        env = os.environ.copy()
        env["SNAP"] = self.model_dir
        env["RAM_GB"] = str(self.ram_gb)
        env["SERVE"] = "1"
        env["NGEN"] = str(self.max_tokens)
        env["TEMP"] = "0.7"
        env["NUCLEUS"] = "0.9"
        env["TOPK"] = "20"
        env["CTX"] = str(self.ctx)
        if self.use_metal:
            env["COLI_METAL"] = "1"
        if self.kv_i8:
            env["KV_I8"] = "1"
        
        self.process = subprocess.Popen(
            [self.engine_path, str(self.cap), "4"],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        
        # Wait for READY (skip header lines)
        while True:
            line = self.process.stdout.readline()
            if not line:
                stderr_output = self.process.stderr.read().decode('utf-8', errors='replace')
                raise RuntimeError(f"Engine exited during startup: {stderr_output}")
            if b"READY" in line:
                break
        # Read STAT line
        self.process.stdout.readline()
        sys.stderr.write("[engine] Ready and waiting for requests\n")
        sys.stderr.flush()
    
    def _send_reset(self):
        """Send RESET command to clear conversation."""
        self.process.stdin.write(b"\x02RESET\n")
        self.process.stdin.flush()
        self._read_until_end()
    
    def _read_until_end(self):
        """Read output until END marker, consuming the STAT line too."""
        output = b""
        end_marker = b"\x01\x01END\x01\x01"
        while True:
            byte = self.process.stdout.read(1)
            if not byte:
                break
            output += byte
            if end_marker in output:
                idx = output.index(end_marker)
                # Consume the newline after END marker
                self.process.stdout.read(1)  # consume \n
                # Read STAT line
                stat = self.process.stdout.readline()
                return output[:idx].decode("utf-8", errors="replace"), stat
        stat = self.process.stdout.readline()
        return output.decode("utf-8", errors="replace"), stat
    
    def _send_prompt(self, prompt, max_tokens, temperature, top_p):
        """Send a PROMPT command and read the response."""
        payload = prompt.encode("utf-8")
        ngen = min(max_tokens, self.max_tokens)
        header = f"\x02PROMPT {len(payload)} {ngen} {temperature:.6f} {top_p:.6f} 0\n".encode()
        
        self.process.stdin.write(header)
        self.process.stdin.write(payload)
        self.process.stdin.write(b"\n")
        self.process.stdin.flush()
        
        # Read output until END marker
        text, stat = self._read_until_end()
        
        # Parse stats
        stat_str = stat.decode("utf-8", errors="replace").strip() if isinstance(stat, bytes) else stat.strip()
        stats = self._parse_stat(stat_str)
        
        return text.strip(), stats
    
    def _parse_stat(self, stat_line):
        """Parse STAT line: STAT <prod> <tok/s> <hit%> <rss_gb> <prompt_tokens> <length_limited> <decode_tps>"""
        parts = stat_line.split()
        if len(parts) >= 7:
            return {
                "tokens": int(parts[1]),
                "tok_per_sec": float(parts[2]),
                "hit_rate": float(parts[3]),
                "rss_gb": float(parts[4]),
                "prompt_tokens": int(parts[5]),
                "length_limited": bool(int(parts[6])),
                "decode_tps": float(parts[7]) if len(parts) > 7 else 0.0,
            }
        return {}
    
    def _run_engine(self, prompt, max_tokens, temperature, top_p):
        """Run the engine with given parameters and return output."""
        enc = self.tokenizer.encode(prompt, add_special_tokens=True)
        token_ids = [2] + enc.ids  # Ensure BOS token
        ids_str = ",".join(str(t) for t in token_ids)
        
        env = os.environ.copy()
        env["SNAP"] = self.model_dir
        env["RAM_GB"] = str(self.ram_gb)
        env["PROMPT"] = "x"
        env["IDS"] = ids_str
        env["NGEN"] = str(min(max_tokens, self.max_tokens))
        env["TEMP"] = str(temperature)
        env["NUCLEUS"] = str(top_p)
        env["TOPK"] = "20"
        env["CTX"] = str(self.ctx)
        if self.use_metal:
            env["COLI_METAL"] = "1"
        if self.kv_i8:
            env["KV_I8"] = "1"
        
        result = subprocess.run(
            [self.engine_path, str(self.cap), "4"],
            env=env,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        return result.stdout
    
    def generate(self, prompt, max_tokens=128, temperature=0.7, top_p=0.9, top_k=20):
        """Generate text from a prompt (non-streaming)."""
        with self.lock:
            start_time = time.time()
            
            if self.persistent:
                # Use serve mode for KV cache persistence
                text, stats = self._send_prompt(prompt, max_tokens, temperature, top_p)
                elapsed = time.time() - start_time
                
                sys.stderr.write(f"[generate] {stats.get('tokens', 0)} tokens in {elapsed:.1f}s ({stats.get('tok_per_sec', 0):.2f} tok/s) | hit rate: {stats.get('hit_rate', 0):.1f}%\n")
                sys.stderr.flush()
                return text
            else:
                # Use IDS mode (new process per request)
                output = self._run_engine(prompt, max_tokens, temperature, top_p)
                elapsed = time.time() - start_time
                
                # Parse output
                token_match = re.search(r'tokens:\s+(.+)', output)
                if token_match:
                    token_ids_str = token_match.group(1).strip()
                    generated_ids = [int(x) for x in token_ids_str.split()]
                    decoded = self.tokenizer.decode(generated_ids)
                    
                    # Extract stats
                    speed_match = re.search(r'([\d.]+) tok/s', output)
                    speed = float(speed_match.group(1)) if speed_match else 0
                    hit_match = re.search(r'expert hit rate: ([\d.]+)%', output)
                    hit_rate = float(hit_match.group(1)) if hit_match else 0
                    
                    sys.stderr.write(f"[generate] {len(generated_ids)} tokens in {elapsed:.1f}s ({speed:.2f} tok/s) | hit rate: {hit_rate:.1f}%\n")
                    sys.stderr.flush()
                    
                    return decoded
                
                sys.stderr.write(f"[generate] Error: {output}\n")
                sys.stderr.flush()
                return output.strip()
    
    def generate_stream(self, prompt, max_tokens=128, temperature=0.7, top_p=0.9, top_k=20):
        """Generate text from a prompt (streaming). Yields text chunks."""
        with self.lock:
            start_time = time.time()
            
            if self.persistent:
                # Use serve mode for KV cache persistence
                text, stats = self._send_prompt(prompt, max_tokens, temperature, top_p)
                elapsed = time.time() - start_time
                
                sys.stderr.write(f"[stream] {stats.get('tokens', 0)} tokens in {elapsed:.1f}s ({stats.get('tok_per_sec', 0):.2f} tok/s) | hit rate: {stats.get('hit_rate', 0):.1f}%\n")
                sys.stderr.flush()
                
                yield text
            else:
                # Use IDS mode (new process per request)
                output = self._run_engine(prompt, max_tokens, temperature, top_p)
                elapsed = time.time() - start_time
                
                # Parse output
                token_match = re.search(r'tokens:\s+(.+)', output)
                if token_match:
                    token_ids_str = token_match.group(1).strip()
                    generated_ids = [int(x) for x in token_ids_str.split()]
                    
                    # Extract stats
                    speed_match = re.search(r'([\d.]+) tok/s', output)
                    speed = float(speed_match.group(1)) if speed_match else 0
                    hit_match = re.search(r'expert hit rate: ([\d.]+)%', output)
                    hit_rate = float(hit_match.group(1)) if hit_match else 0
                    
                    sys.stderr.write(f"[stream] {len(generated_ids)} tokens in {elapsed:.1f}s ({speed:.2f} tok/s) | hit rate: {hit_rate:.1f}%\n")
                    sys.stderr.flush()
                    
                    # Decode each token individually for streaming
                    for tid in generated_ids:
                        token_text = self.tokenizer.decode([tid])
                        yield token_text
                else:
                    sys.stderr.write(f"[stream] Error: {output}\n")
                    sys.stderr.flush()
                    yield output.strip()
    
    def get_health(self):
        """Get engine health information."""
        return {
            "status": "ok",
            "scheduler": {"active": True, "capacity": 1, "queued": 0, "max_queue": 8,
                         "queue_timeout_seconds": 300, "admitted": 0, "completed": 0,
                         "rejected": 0, "timed_out": 0, "cancelled": 0},
            "kv_slots": 1,
            "tiers": {"vram": 0, "ram": self.ram_gb, "disk": 56, "vram_gb": 0, "ram_gb": self.ram_gb},
            "hwinfo": {"cores": os.cpu_count() or 8, "ram_total_gb": 25, "ram_avail_gb": 15,
                      "gpus": 1 if self.use_metal else 0, "vram_total_gb": 0, "cpu": "Apple Silicon", "gpu": "Apple GPU" if self.use_metal else ""}
        }
    
    def close(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except:
                self.process.kill()


class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stderr.write(f"[api] {self.address_string()} - {format % args}\n")
    
    def send_json(self, status, data, extra_headers=None):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)
    
    def send_sse(self, status, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()
    
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/v1/models":
            self.send_json(200, {
                "object": "list",
                "data": [{
                    "id": "laguna-s-2.1",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "poolside"
                }]
            })
        elif path == "/health":
            health = self.server.engine.get_health()
            self.send_json(200, health)
        elif path == "/profile":
            self.send_json(200, {"seq": 0, "turns": []})
        elif path == "/experts":
            self.send_json(200, {"rows": 48, "cols": 256, "map": ""})
        elif path == "/experts.json":
            self.send_json(200, {"rows": 48, "cols": 256, "map": ""})
        else:
            self.send_json(404, {"error": {"message": "Not found"}})
    
    def do_POST(self):
        path = urlparse(self.path).path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_json(400, {"error": {"message": "Invalid JSON"}})
            return
        
        if path == "/v1/chat/completions":
            self.handle_chat_completion(data)
        elif path == "/v1/completions":
            self.handle_completion(data)
        else:
            self.send_json(404, {"error": {"message": "Not found"}})
    
    def handle_chat_completion(self, body):
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", body.get("max_completion_tokens", 128))
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        stream = body.get("stream", False)
        
        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt = "\n".join(prompt_parts)
        
        engine = self.server.engine
        request_id = f"chatcmpl-{int(time.time())}"
        
        if stream:
            self.handle_streaming_response(request_id, "chat.completion.chunk", prompt, max_tokens, temperature, top_p, is_chat=True)
        else:
            try:
                output = engine.generate(prompt, max_tokens, temperature, top_p)
                self.send_json(200, {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "laguna-s-2.1",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": output},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }, extra_headers={"x-request-id": request_id, "x-colibri-queue-wait-ms": "0"})
            except Exception as e:
                self.send_json(500, {"error": {"message": str(e)}})
    
    def handle_completion(self, body):
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 128)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        stream = body.get("stream", False)
        
        engine = self.server.engine
        request_id = f"cmpl-{int(time.time())}"
        
        if stream:
            self.handle_streaming_response(request_id, "text_completion", prompt, max_tokens, temperature, top_p, is_chat=False)
        else:
            try:
                output = engine.generate(prompt, max_tokens, temperature, top_p)
                self.send_json(200, {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": "laguna-s-2.1",
                    "choices": [{
                        "index": 0,
                        "text": output,
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }, extra_headers={"x-request-id": request_id, "x-colibri-queue-wait-ms": "0"})
            except Exception as e:
                self.send_json(500, {"error": {"message": str(e)}})
    
    def handle_streaming_response(self, request_id, object_name, prompt, max_tokens, temperature, top_p, is_chat, enable_thinking=False):
        engine = self.server.engine
        
        try:
            self.send_sse(200, {
                "x-request-id": request_id,
                "x-colibri-queue-wait-ms": "0"
            })
            
            # Send initial chunk
            if is_chat:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "laguna-s-2.1",
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None
                    }]
                }
            else:
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": "laguna-s-2.1",
                    "choices": [{
                        "index": 0,
                        "text": "",
                        "finish_reason": None
                    }]
                }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            
            # Stream tokens
            for token_text in engine.generate_stream(prompt, max_tokens, temperature, top_p):
                if is_chat:
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "laguna-s-2.1",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None
                        }]
                    }
                else:
                    chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": "laguna-s-2.1",
                        "choices": [{
                            "index": 0,
                            "text": token_text,
                            "finish_reason": None
                        }]
                    }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            
            # Send final chunk
            if is_chat:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "laguna-s-2.1",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
            else:
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": "laguna-s-2.1",
                    "choices": [{
                        "index": 0,
                        "text": "",
                        "finish_reason": "stop"
                    }]
                }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            
        except Exception as e:
            sys.stderr.write(f"[api] Error: {e}\n")
            try:
                error_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk" if is_chat else "text_completion",
                    "created": int(time.time()),
                    "model": "laguna-s-2.1",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "error"
                    }]
                }
                self.wfile.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except:
                pass


class LagunaAPIServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, addr, engine, **kwargs):
        super().__init__(addr, APIHandler)
        self.engine = engine


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible API server for Laguna S 2.1")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--engine", default=None, help="Path to laguna binary")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cap", type=int, default=64, help="Expert cache size (default: 64)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--ram-gb", type=int, default=20, help="RAM budget in GB (default: 20)")
    parser.add_argument("--ctx", type=int, default=1024, help="Max context length (default: 1024)")
    parser.add_argument("--warm-cache", action="store_true", 
                        help="Pre-load model files into OS file cache for faster startup")
    parser.add_argument("--metal", action="store_true",
                        help="Enable Metal GPU acceleration")
    parser.add_argument("--kv-i8", action="store_true", default=True,
                        help="Use int8 KV cache compression (4x memory savings, default: on)")
    parser.add_argument("--persistent", action="store_true",
                        help="Keep engine process alive between requests (enables KV cache reuse)")
    args = parser.parse_args()
    
    if args.engine is None:
        args.engine = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laguna")
    
    print(f"Starting Laguna S 2.1 API server...", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Engine: {args.engine}", file=sys.stderr)
    if args.metal:
        print(f"Metal backend: enabled", file=sys.stderr)
    if args.warm_cache:
        print(f"Warm cache mode: enabled (pre-loading model files)", file=sys.stderr)
    if args.persistent:
        print(f"Persistent mode: enabled (KV cache preserved between requests)", file=sys.stderr)
    
    engine = LagunaEngine(args.model, args.engine, args.cap, args.max_tokens, args.ram_gb, args.warm_cache, args.metal, args.ctx, args.kv_i8, args.persistent)
    server = LagunaAPIServer((args.host, args.port), engine)
    
    print(f"API listening on http://{args.host}:{args.port}/v1", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", file=sys.stderr)
    finally:
        engine.close()
        server.server_close()

if __name__ == "__main__":
    main()
