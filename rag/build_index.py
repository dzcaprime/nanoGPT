import argparse
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.index import build_index, save_index


def parse_args():
    parser = argparse.ArgumentParser(description="Build a local lexical RAG index.")
    parser.add_argument("--corpus", nargs="+", required=True, help="Files or directories to index.")
    parser.add_argument("--out_dir", default="rag_index", help="Directory for index.json.")
    parser.add_argument("--chunk_chars", type=int, default=1200)
    parser.add_argument("--chunk_overlap", type=int, default=150)
    return parser.parse_args()


def main():
    args = parse_args()
    index = build_index(args.corpus, args.chunk_chars, args.chunk_overlap)
    index_path = save_index(index, args.out_dir)
    print(f"Wrote {index['num_chunks']} chunks to {index_path}")


if __name__ == "__main__":
    main()
