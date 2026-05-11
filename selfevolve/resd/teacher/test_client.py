"""Quick test client for the teacher worker via ZMQ proxy."""

import argparse
import io

import torch
import zmq
from transformers import AutoTokenizer


def serialize(data):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return buffer.getbuffer()


def deserialize(message):
    buffer = io.BytesIO(message)
    return torch.load(buffer)


def main():
    parser = argparse.ArgumentParser(description="Test teacher worker output")
    parser.add_argument("--proxy-addr", type=str, default="127.0.0.1:15555",
                        help="Proxy frontend address (default: 127.0.0.1:15555)")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-4B",
                        help="Path to tokenizer (same as model path)")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                        help="Text prompt to send")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--only-response", action="store_true",
                        help="Only return response logprobs (skip prompt logprobs)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    prompt_token_ids = tokenizer.encode(args.prompt)
    print(f"Prompt: {args.prompt}")
    print(f"Token IDs ({len(prompt_token_ids)} tokens): {prompt_token_ids[:20]}...")

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{args.proxy_addr}")

    request = {
        "prompt_token_ids": [prompt_token_ids],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "only_response": args.only_response,
    }

    print(f"\nSending request to {args.proxy_addr}...")
    socket.send(serialize(request))
    reply = deserialize(socket.recv())

    if reply.get("status") != "ok":
        print(f"Error: {reply}")
        return

    responses = reply["responses"]
    logprobs = reply.get("teacher_topk_logprobs", [])
    indices = reply.get("teacher_topk_indices", [])

    for i, resp_ids in enumerate(responses):
        text = tokenizer.decode(resp_ids, skip_special_tokens=True)
        print(f"\n--- Response {i} ---")
        print(f"Token IDs: {resp_ids}")
        print(f"Text: {text}")
        if i < len(logprobs) and len(logprobs[i]) > 0:
            print(f"Logprobs shape: {logprobs[i].shape}")
            print(f"Top-k indices shape: {indices[i].shape}")

    socket.close()
    context.term()


if __name__ == "__main__":
    main()
