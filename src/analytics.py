"""
Módulo de Analítica, Métricas de Satisfacción y Detección de Vacíos Normativos para la Universidad Nacional del Valle Alto (UNVA).
"""
import os
import io
import csv
import json
import uuid
import threading
import tempfile
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

VZLA_TZ = timezone(timedelta(hours=-4))  # Hora oficial de Venezuela (VET / UTC-4)

try:
    from src.config import DATA_DIR
except ImportError:
    from config import DATA_DIR

ANALYTICS_FILE = DATA_DIR / "analytics.json"
_ANALYTICS_LOCK = threading.Lock()


def _load_data() -> Dict[str, Any]:
    if not ANALYTICS_FILE.exists():
        initial = {"queries": []}
        _save_data(initial)
        return initial
    try:
        with open(ANALYTICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"queries": []}


def _save_data(data: Dict[str, Any]):
    """Escritura atómica para evitar corrupción de analytics.json bajo concurrencia."""
    try:
        ANALYTICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(ANALYTICS_FILE.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, ANALYTICS_FILE)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    except Exception as e:
        print(f"[!] Error guardando analytics.json: {e}")


def _normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _classify_topic(query: str) -> str:
    q = _normalize(query)
    if any(k in q for k in ["cashea", "cuota", "pago", "pagar", "arancel", "costo", "precio", "banco", "transferencia", "mensualidad", "matricula"]):
        return "Pagos y Cashea"
    if any(k in q for k in ["sede", "guatire", "caracas", "barcelona", "puerto la cruz", "anzoategui", "ubicacion", "telefono", "direccion"]):
        return "Sedes y Contacto"
    if any(k in q for k in ["nota", "pi", "inasistencia", "corte", "aprobar", "aprobatoria", "evaluacion", "examen", "raspar", "promedio"]):
        return "Notas y Reglamentos (PI)"
    if any(k in q for k in ["ingreso", "bachiller", "requisito", "recaudo", "carpeta", "inscripcion", "inscribir", "ternanet", "admision"]):
        return "Admisión y TernaNet"
    if any(k in q for k in ["servicio comunitario", "comunitario", "pasantia", "grado", "solvencia", "titulo", "tesis"]):
        return "Servicio Comunitario y Grado"
    if any(k in q for k in ["carrera", "informatica", "administracion", "contaduria", "mercadeo", "publicidad", "turno", "sabatino", "nocturno", "horario"]):
        return "Oferta Académica y Turnos"
    return "Otras Consultas"


def _is_knowledge_gap(answer: str, sources: List[str], is_gap_override: Optional[bool] = None) -> bool:
    """Detecta si la respuesta del asistente indica un vacío normativo real."""
    if is_gap_override is not None:
        return is_gap_override
    ans = _normalize(answer)
    indicators = [
        "no encontre informacion relevante",
        "no dispongo de informacion oficial",
        "no se encuentra en las normativas",
        "no se especifica en los reglamentos",
        "no aparece en los documentos",
        "te sugiero consultar directamente en control de estudios",
        "no cuento con esa informacion",
        "no tengo registro de",
        "no esta disponible en las fuentes oficiales"
    ]
    if any(ind in ans for ind in indicators):
        return True
    if not sources or len(sources) == 0:
        # Evitar falsos vacíos en saludos o respuestas de expediente TernaNet
        if any(ok in ans for ok in ["modo general", "modo personalizado", "con mucho gusto", "expediente", "ternanet"]):
            return False
        return True
    return False


def record_query(
    query: str,
    channel: str,
    answer: str,
    sources: List[str],
    is_gap_override: Optional[bool] = None
) -> str:
    """Registra una consulta en la base de datos de analítica."""
    query_id = "q_" + uuid.uuid4().hex[:10]
    topic = _classify_topic(query)
    is_gap = _is_knowledge_gap(answer, sources, is_gap_override=is_gap_override)
    now = datetime.now(VZLA_TZ).strftime("%d/%m/%Y %I:%M:%S %p")

    entry = {
        "id": query_id,
        "timestamp": now,
        "channel": channel,
        "query": query.strip(),
        "topic": topic,
        "is_gap": is_gap,
        "sources": sources,
        "rating": None,
        "comment": None
    }

    with _ANALYTICS_LOCK:
        data = _load_data()
        data.setdefault("queries", [])
        data["queries"].append(entry)

        # Mantener máximo 500 registros para optimizar memoria y disco
        if len(data["queries"]) > 500:
            data["queries"] = data["queries"][-500:]

        _save_data(data)
    return query_id


def record_feedback(query_id: str, rating: str, comment: Optional[str] = None) -> bool:
    """Registra calificación (up o down) para una consulta dada."""
    with _ANALYTICS_LOCK:
        data = _load_data()
        queries = data.get("queries", [])
        for q in reversed(queries):
            if q.get("id") == query_id:
                q["rating"] = rating
                if comment:
                    q["comment"] = comment.strip()
                _save_data(data)
                return True
    return False


def get_analytics_summary() -> Dict[str, Any]:
    """Genera métricas consolidadas para el panel de administración."""
    data = _load_data()
    queries = data.get("queries", [])
    total = len(queries)

    web_count = sum(1 for q in queries if q.get("channel") == "web")
    tg_count = sum(1 for q in queries if q.get("channel") == "telegram")

    rated = [q for q in queries if q.get("rating")]
    up_count = sum(1 for q in rated if q.get("rating") == "up")
    down_count = sum(1 for q in rated if q.get("rating") == "down")
    satisfaction_rate = round((up_count / len(rated) * 100), 1) if rated else 100.0

    # Conteo por temas
    topics_count: Dict[str, int] = {}
    for q in queries:
        t = q.get("topic", "Otras Consultas")
        topics_count[t] = topics_count.get(t, 0) + 1

    sorted_topics = [
        {"topic": k, "count": v, "percent": round((v / total * 100), 1) if total else 0}
        for k, v in sorted(topics_count.items(), key=lambda x: x[1], reverse=True)
    ]

    # Vacíos informativos detectados
    gaps = [
        {
            "id": q.get("id"),
            "timestamp": q.get("timestamp"),
            "query": q.get("query"),
            "channel": q.get("channel"),
            "topic": q.get("topic")
        }
        for q in reversed(queries) if q.get("is_gap")
    ][:20]

    # Consultas recientes
    recent = [
        {
            "id": q.get("id"),
            "timestamp": q.get("timestamp"),
            "query": q.get("query"),
            "channel": q.get("channel"),
            "topic": q.get("topic"),
            "rating": q.get("rating"),
            "is_gap": q.get("is_gap")
        }
        for q in reversed(queries)
    ][:30]

    return {
        "total_queries": total,
        "channels": {
            "web": web_count,
            "telegram": tg_count
        },
        "satisfaction": {
            "rate": satisfaction_rate,
            "total_rated": len(rated),
            "up": up_count,
            "down": down_count
        },
        "topics": sorted_topics,
        "gaps": gaps,
        "total_gaps": len([q for q in queries if q.get("is_gap")]),
        "recent_queries": recent
    }


def _safe_csv_cell(value: str) -> str:
    """Previene CSV/Formula Injection al abrir el archivo en Excel y limpia saltos de línea."""
    text = str(value or "").replace("\r\n", " ").replace("\n", " ").strip()
    if text and text[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + text
    return text


def export_csv() -> str:
    """Genera un archivo CSV seguro (RFC 4180 + anti-CSV-injection) con el historial de consultas."""
    data = _load_data()
    queries = data.get("queries", [])
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["ID", "Fecha", "Canal", "Tema", "Consulta", "Vacio_Normativo", "Calificacion", "Comentario"])
    for q in queries:
        writer.writerow([
            _safe_csv_cell(q.get("id", "")),
            _safe_csv_cell(q.get("timestamp", "")),
            _safe_csv_cell(q.get("channel", "")),
            _safe_csv_cell(q.get("topic", "")),
            _safe_csv_cell(q.get("query", "")),
            "SI" if q.get("is_gap") else "NO",
            _safe_csv_cell(q.get("rating") or "Sin calificar"),
            _safe_csv_cell(q.get("comment") or "")
        ])
    return buf.getvalue()


def reset_analytics():
    """Borra el historial de analítica."""
    with _ANALYTICS_LOCK:
        _save_data({"queries": []})
