"""
RAG System Execution Telemetry & Performance Metrics Logger
"""
import time
import json
import logging

logger = logging.getLogger("rag_telemetry")

class TelemetryLogger:
    @staticmethod
    def log_retrieval(query: str, session_id: str, retrieved_count: int, latency_ms: float):
        payload = {
            "event": "retrieval_executed",
            "session_id": session_id,
            "query": query,
            "chunks_retrieved": retrieved_count,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time()
        }
        logger.info(json.dumps(payload))

    @staticmethod
    def log_eval(answer_id: str, faithfulness: float, relevancy: float, decision: str):
        payload = {
            "event": "ragas_eval_completed",
            "answer_id": answer_id,
            "faithfulness": round(faithfulness, 4),
            "answer_relevancy": round(relevancy, 4),
            "decision": decision,
            "timestamp": time.time()
        }
        logger.info(json.dumps(payload))
