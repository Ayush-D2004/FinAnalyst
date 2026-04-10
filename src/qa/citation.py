from typing import List, Dict, Any

class CitationManager:
    """
    Handles formatting of citations and tracking mapping 
    from generated indices back to physical SQLite chunks.
    """
    @staticmethod
    def attach_citations(answer: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """
        Takes the raw generated answer which includes '[Doc 1]', '[Doc 2]' 
        and extracts the referenced chunks to serve in the UI.
        """
        citations = []
        for i, chunk in enumerate(retrieved_chunks):
            doc_ref = f"[Doc {i+1}]"
            # simple check if the doc_ref is in the text
            if doc_ref in answer:
                citations.append({
                    "ref": doc_ref,
                    "section": chunk.get("section_name", "General"),
                    "text": chunk.get("chunk_text", "")
                })
                
        return {
            "answer": answer,
            "citations": citations,
            "raw_context": retrieved_chunks
        }
