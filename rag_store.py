"""RAG 저장소 - 임베딩 생성/저장/검색 (sqlite3 + Ollama embed API)"""

import json
import math
import os
import sqlite3
import time
import urllib.request

EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE = "http://localhost:11434"
DB_DIR = "tmp/rag"
DB_PATH = os.path.join(DB_DIR, "vectors.db")

CHUNK_MAX_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200


class RAGStore:
    def __init__(self, db_path=DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (
                session TEXT PRIMARY KEY,
                indexed_turns INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_session
                ON chunks(session);
        """)
        self.conn.commit()

    def embed(self, text):
        """Ollama /api/embed로 텍스트를 벡터로 변환한다."""
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/embed",
            data=json.dumps({
                "model": EMBED_MODEL,
                "input": text
            }).encode(),
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        return result["embeddings"][0]

    def add_turn(self, session, role, content, turn_index):
        """대화 턴을 청킹하여 임베딩 후 저장한다."""
        chunks = self._chunk_text(content)
        now = time.time()
        for chunk in chunks:
            vec = self.embed(chunk)
            self.conn.execute(
                "INSERT INTO chunks (session, role, content, embedding, turn_index, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session, role, chunk, json.dumps(vec), turn_index, now)
            )
        self.conn.commit()

    def add_pair(self, session, user_msg, assistant_msg, turn_index):
        """user+assistant 쌍을 하나의 청크 단위로 저장한다."""
        combined = f"질문: {user_msg}\n답변: {assistant_msg}"
        chunks = self._chunk_text(combined)
        now = time.time()
        for chunk in chunks:
            vec = self.embed(chunk)
            self.conn.execute(
                "INSERT INTO chunks (session, role, content, embedding, turn_index, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session, "pair", chunk, json.dumps(vec), turn_index, now)
            )
        self.conn.commit()

    def search(self, query, session, top_k=10):
        """쿼리와 가장 유사한 청크 top_k개를 반환한다."""
        query_vec = self.embed(query)

        rows = self.conn.execute(
            "SELECT content, embedding, turn_index, role FROM chunks WHERE session = ?",
            (session,)
        ).fetchall()

        if not rows:
            return []

        scored = []
        for row in rows:
            vec = json.loads(row["embedding"])
            sim = self._cosine_similarity(query_vec, vec)
            scored.append({
                "content": row["content"],
                "turn_index": row["turn_index"],
                "role": row["role"],
                "score": sim,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_indexed_turns(self, session):
        """세션의 인덱싱된 턴 수를 반환한다."""
        row = self.conn.execute(
            "SELECT indexed_turns FROM meta WHERE session = ?", (session,)
        ).fetchone()
        return row["indexed_turns"] if row else 0

    def set_indexed_turns(self, session, count):
        """세션의 인덱싱된 턴 수를 업데이트한다."""
        self.conn.execute(
            "INSERT INTO meta (session, indexed_turns) VALUES (?, ?) "
            "ON CONFLICT(session) DO UPDATE SET indexed_turns = ?",
            (session, count, count)
        )
        self.conn.commit()

    def migrate_from_messages(self, session, messages):
        """JSON 메시지 배열을 벡터 DB로 인덱싱한다."""
        indexed = self.get_indexed_turns(session)

        # system 메시지 제외, user/assistant 쌍으로 묶기
        turns = []
        i = 0
        while i < len(messages):
            if messages[i]["role"] == "system":
                i += 1
                continue
            if messages[i]["role"] == "user":
                user_msg = messages[i]["content"]
                assistant_msg = ""
                if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                    assistant_msg = messages[i + 1]["content"]
                    i += 1
                turns.append((user_msg, assistant_msg))
            i += 1

        # 이미 인덱싱된 턴은 건너뛰기
        new_turns = turns[indexed:]
        if not new_turns:
            return 0

        for idx, (user_msg, assistant_msg) in enumerate(new_turns):
            turn_index = indexed + idx
            self.add_pair(session, user_msg, assistant_msg, turn_index)

        total = indexed + len(new_turns)
        self.set_indexed_turns(session, total)
        return len(new_turns)

    def _chunk_text(self, text):
        """긴 텍스트를 청크로 분할한다."""
        if len(text) <= CHUNK_MAX_CHARS:
            return [text]

        # 단락 기준 분할
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 > CHUNK_MAX_CHARS and current:
                chunks.append(current.strip())
                # 오버랩: 이전 청크의 마지막 부분을 다음 청크에 포함
                overlap = current[-CHUNK_OVERLAP_CHARS:] if len(current) > CHUNK_OVERLAP_CHARS else current
                current = overlap + "\n\n" + para
            else:
                current = current + "\n\n" + para if current else para

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text]

    @staticmethod
    def _cosine_similarity(a, b):
        """두 벡터의 코사인 유사도를 계산한다."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
