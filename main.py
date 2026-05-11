"""CLI demo for Memory-aware Profile Rewriting."""

from __future__ import annotations

import argparse
import json

from profile_rewriter import MemoryAwareProfileRewriter

EXIT_COMMANDS = {"exit", "quit", "q"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Memory-aware Profile Rewriting demo powered by Qwen3-8B."
    )
    parser.add_argument(
        "--user_profile_path",
        default="user_profile.txt",
        help="Path to local user_profile txt file. Defaults to user_profile.txt.",
    )
    parser.add_argument(
        "--queries",
        nargs="*",
        help="Process multiple queries sequentially with one shared rewriter instance.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive multi-turn mode.",
    )
    return parser.parse_args()


def print_result(result: dict) -> None:
    print("\n========== 当前轮结构化结果 ==========")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n========== 最终 rewritten profile ==========")
    print(result.get("rewritten_profile", "INVALID"))
    print("==========================================\n")


def print_final_memory(memory: list) -> None:
    print("\n========== 最终 session memory ==========")
    print(json.dumps(memory, ensure_ascii=False, indent=2))
    print("========================================\n")


def run_interactive_loop(rewriter: MemoryAwareProfileRewriter) -> None:
    print("进入交互式模式。输入 exit / quit / q 退出。")
    while True:
        try:
            query = input("请输入 query: ").strip()
            if query.lower() in EXIT_COMMANDS:
                break
            if not query:
                continue
            result = rewriter.process_query(query)
            print_result(result)
        except KeyboardInterrupt:
            print("\n检测到手动中断，准备退出。")
            break
    print_final_memory(rewriter.memory)


def main() -> None:
    args = parse_args()
    rewriter = MemoryAwareProfileRewriter(user_profile_path=args.user_profile_path)

    if args.queries:
        for query in args.queries:
            result = rewriter.process_query(query)
            print_result(result)
        print_final_memory(rewriter.memory)
        return

    run_interactive_loop(rewriter)


if __name__ == "__main__":
    main()
