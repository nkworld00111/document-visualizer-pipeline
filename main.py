import os
import sys
import json
import webbrowser
from folder_extractor import extract_documents_from_paths
from embedder import embed_documents
from visualizer import create_visualizations
from weaviate_utils import connect_weaviate, ensure_schema, upsert_documents, close_connection

def main(paths):
    print(f" Starting pipeline for {len(paths)} path(s)...\n")
    print(" Step 1 — Extracting document(s)...")
    docs = extract_documents_from_paths(paths)
    print(f" Extracted {len(docs)} document(s).\n")

    if not docs:
        print("No supported documents found. Exiting.")
        return

    print(" Step 2 — Generating embeddings...")
    docs = embed_documents(docs)
    print(" Embeddings generated successfully.\n")

    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(os.getcwd(), f"extracted_with_embeddings_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"Saved extracted + embedded docs to: {output_file}\n")

    print("Step 3 — Connecting to Weaviate (optional)...")
    client = connect_weaviate(port=8093)  
    if client:
        ensure_schema(client, class_name="Document")
        try:
            upsert_documents(client, docs, class_name="Document")
            print("Documents inserted/updated successfully in Weaviate.")
        except Exception as e:
            print(f"Failed to upsert documents to Weaviate: {e}")
        finally:
            close_connection(client)
    else:
        print("Weaviate connection not available — skipping storage step.\n")

    print("Step 4 — Creating visualization dashboard...")
    try:
        res = create_visualizations(docs)
        html_path = res.get("dashboard", "") if isinstance(res, dict) else res
        if not html_path or not isinstance(html_path, str):
            raise ValueError("Invalid dashboard path returned from visualizer.")
        print(f"Dashboard created at: {html_path}")
        abs_path = os.path.abspath(html_path)
        if os.path.exists(abs_path):
            print(f"Opening dashboard in your browser: {abs_path}")
            webbrowser.open(f"file://{abs_path}")
        else:
            print(f"Dashboard file not found: {abs_path}")
    except Exception as e:
        print(f"Visualization failed: {e}")

    print("\n Pipeline complete — All stages executed (with best-effort).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <folder_path1> [folder_path2 ...]")
        sys.exit(1)
    main(sys.argv[1:])
