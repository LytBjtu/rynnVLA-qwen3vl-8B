import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import List

import requests

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PlanningCliClient:
    def __init__(self, server_url: str, llm_type: str, timeout: int = 300) -> None:
        self.server_url = server_url.rstrip("/")
        self.llm_type = llm_type
        self.timeout = timeout

        self.task_prompt = ""
        self.current_round_images: List[str] = []
        self.history_image_paths: List[str] = []
        self.conversation = []

    def load_model(self) -> None:
        url = f"{self.server_url}/loading/"
        data = {"llm_type": self.llm_type}
        response = requests.post(url, data=data, timeout=self.timeout)
        response.raise_for_status()
        print(response.json().get("message", "Model loaded."))

    def set_plan(self, plan_text: str) -> None:
        self.task_prompt = plan_text.strip()
        if self.task_prompt:
            print("Plan updated.")
        else:
            print("Plan cleared.")

    def upload_images(self, entries: List[str], clear_existing: bool = False) -> None:
        image_paths = self._collect_images(entries)
        if not image_paths:
            print("No valid images found.")
            return

        if clear_existing:
            self.current_round_images = image_paths
        else:
            self.current_round_images.extend(image_paths)

        print(f"Queued {len(image_paths)} image(s) for this round.")

    def generate(self) -> None:
        if not self.current_round_images:
            raise ValueError("No images queued. Use upload first.")

        user_msg = self._build_user_message(self.current_round_images)
        self.conversation.append(user_msg)
        self.history_image_paths.extend(self.current_round_images)
        self.current_round_images = []

        infer_url = f"{self.server_url}/inference/"
        payload = {
            "llm_type": self.llm_type,
            "conversation": json.dumps(self.conversation, ensure_ascii=False),
        }

        files_to_send = []
        opened_files = []

        try:
            for image_path in self.history_image_paths:
                filename = os.path.basename(image_path)
                content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                fp = open(image_path, "rb")
                opened_files.append(fp)
                files_to_send.append(("video_files", (filename, fp, content_type)))

            response = requests.post(
                infer_url,
                data=payload,
                files=files_to_send,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json().get("result", {})
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {"outputs": result, "frame_idx": -1}

            outputs = result.get("outputs", "")
            frame_idx = result.get("frame_idx", -1)

            self.conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": outputs}],
                }
            )

            print("\n=== Planning Result ===")
            print(f"frame_idx: {frame_idx}")
            print(outputs)
            print("=======================\n")

        finally:
            for fp in opened_files:
                fp.close()

    def reset(self) -> None:
        self.task_prompt = ""
        self.current_round_images = []
        self.history_image_paths = []
        self.conversation = []
        print("Conversation and state reset.")

    def show_state(self) -> None:
        print("\n--- Client State ---")
        print(f"Model: {self.llm_type}")
        print(f"Server: {self.server_url}")
        print(f"Plan set: {'yes' if bool(self.task_prompt) else 'no'}")
        print(f"Queued images: {len(self.current_round_images)}")
        print(f"Uploaded history images: {len(self.history_image_paths)}")
        print(f"Conversation turns: {len(self.conversation)}")
        print("--------------------\n")

    def _build_user_message(self, image_paths: List[str]) -> dict:
        content = []

        is_first_user_round = not any(msg.get("role") == "user" for msg in self.conversation)
        if is_first_user_round:
            if not self.task_prompt:
                raise ValueError("First round requires plan text. Use plan command first.")
            content.append({"type": "text", "text": self.task_prompt})

        for path in image_paths:
            content.append({"type": "image", "image": path})

        return {"role": "user", "content": content}

    @staticmethod
    def _collect_images(entries: List[str]) -> List[str]:
        image_paths: List[str] = []

        for item in entries:
            p = Path(item).expanduser().resolve()
            if p.is_dir():
                dir_images = [
                    str(fp)
                    for fp in sorted(p.iterdir())
                    if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTS
                ]
                image_paths.extend(dir_images)
            elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                image_paths.append(str(p))

        # Deduplicate while preserving order.
        deduped = list(dict.fromkeys(image_paths))
        return deduped


def run_interactive(client: PlanningCliClient) -> None:
    print("Planning CLI started. Type 'help' for commands.")

    while True:
        raw = input("planning> ").strip()
        if not raw:
            continue

        if raw in {"quit", "exit"}:
            print("Bye.")
            return

        if raw == "help":
            print_help()
            continue

        if raw == "load":
            try:
                client.load_model()
            except Exception as exc:
                print(f"Load failed: {exc}")
            continue

        if raw.startswith("plan "):
            client.set_plan(raw[5:].strip())
            continue

        if raw == "plan":
            plan_text = input("Enter plan/task: ").strip()
            client.set_plan(plan_text)
            continue

        if raw.startswith("upload "):
            entries = raw[7:].split()
            client.upload_images(entries)
            continue

        if raw == "upload":
            entries = input("Enter image paths or folders (space separated): ").strip().split()
            client.upload_images(entries)
            continue

        if raw == "generate":
            try:
                client.generate()
            except Exception as exc:
                print(f"Generate failed: {exc}")
            continue

        if raw == "show":
            client.show_state()
            continue

        if raw == "reset":
            client.reset()
            continue

        print("Unknown command. Type 'help' for available commands.")


def print_help() -> None:
    print(
        """
Commands:
  load                Load model on server
  plan <text>         Set first-round task/plan text
  upload <paths...>   Add image files and/or folders for current round
  generate            Send request and print planning result
  show                Show local client state
  reset               Reset conversation and local state
  help                Show this help
  quit/exit           Exit
""".strip()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linux CLI client for planning server")
    parser.add_argument(
        "--server_url",
        type=str,
        required=True,
        help="Base URL of planning server, e.g. http://127.0.0.1:8001",
    )
    parser.add_argument(
        "--llm_type",
        type=str,
        default="rynnbrain_planning",
        help="Model type to use on server",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP timeout in seconds",
    )

    parser.add_argument(
        "--load_model",
        action="store_true",
        help="Load model on startup",
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="",
        help="Set initial task/plan text before generate",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="*",
        default=[],
        help="Image paths or folders to queue before generate",
    )
    parser.add_argument(
        "--generate_once",
        action="store_true",
        help="Run one inference after processing startup options, then exit",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = PlanningCliClient(
        server_url=args.server_url,
        llm_type=args.llm_type,
        timeout=args.timeout,
    )

    if args.load_model:
        client.load_model()

    if args.plan:
        client.set_plan(args.plan)

    if args.images:
        client.upload_images(args.images)

    if args.generate_once:
        client.generate()
        return

    run_interactive(client)


if __name__ == "__main__":
    main()
