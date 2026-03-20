#!/usr/bin/env python3
"""기존 대화 기록(tmp/chat/*.json)을 벡터 DB로 마이그레이션한다."""

import json
import os
import sys

from rag_store import RAGStore

HISTORY_DIR = "tmp/chat"


def migrate():
    if not os.path.exists(HISTORY_DIR):
        print("대화 기록 없음.")
        return

    store = RAGStore()
    files = [f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")]

    if not files:
        print("대화 기록 파일 없음.")
        return

    total_turns = 0
    for fname in sorted(files):
        session = fname[:-5]
        filepath = os.path.join(HISTORY_DIR, fname)

        with open(filepath) as f:
            messages = json.load(f)

        already = store.get_indexed_turns(session)
        print(f"  [{session}] 전체 메시지: {len(messages)}개, 인덱싱 완료: {already}턴", end="")

        new_count = store.migrate_from_messages(session, messages)
        total_turns += new_count

        if new_count > 0:
            print(f" → {new_count}턴 새로 인덱싱")
        else:
            print(" → 최신 상태")

    print(f"\n완료: 총 {total_turns}턴 인덱싱됨")


if __name__ == "__main__":
    print("대화 기록 마이그레이션 시작...")
    migrate()
