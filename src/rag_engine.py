import os
import re
import math
import time
import json
import asyncio
import unicodedata
from collections import Counter
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple

try:
    from src.config import (
        TOP_K_RESULTS, GEMINI_API_KEY,
        GEMINI_MODEL, OPENAI_API_KEY, OPENAI_MODEL
    )
    from src.ternanet import (
        get_student_by_cedula, detect_cedula_in_text,
        format_student_context_for_rag, normalize_cedula
    )
    import src.analytics as analytics
except ImportError:
    from config import (
        TOP_K_RESULTS, GEMINI_API_KEY,
        GEMINI_MODEL, OPENAI_API_KEY, OPENAI_MODEL
    )
    from ternanet import (
        get_student_by_cedula, detect_cedula_in_text,
        format_student_context_for_rag, normalize_cedula
    )
    import analytics

# Try importing modern Google GenAI Client (forzar v1beta para soportar alias *-latest en cualquier versión del SDK)
try:
    from google import genai
    gemini_client = genai.Client(api_key=GEMINI_API_KEY, http_options={"api_version": "v1beta"}) if GEMINI_API_KEY else None
    HAS_GEMINI = True
except ImportError:
    gemini_client = None
    HAS_GEMINI = False

# Try importing OpenAI
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Stopwords en español para indexación y búsqueda BM25
SPANISH_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "en", "por", "para", "con", "sin", "sobre", "entre", "hasta", "desde",
    "que", "como", "cual", "cuales", "donde", "cuando", "cuanto", "cuantos",
    "cuantas", "quien", "quienes", "es", "son", "fue", "era", "esta", "estan",
    "estoy", "hay", "ser", "estar", "tener", "tengo", "puede", "puedo", "pueden",
    "mi", "mis", "me", "te", "se", "su", "sus", "nos", "le", "les", "lo",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "eso",
    "y", "o", "u", "e", "a", "si", "no", "ni", "mas", "pero", "muy", "ya",
    "hola", "buenos", "buenas", "dias", "tardes", "noches", "gracias", "favor",
    "quiero", "saber", "necesito", "informacion", "dime", "ayuda", "consulta"
}


def normalize_text(text: str) -> str:
    """Elimina tildes/diacríticos y convierte a minúsculas para búsqueda insensible a acentos."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


# Mapa de sinónimos estudiantiles venezolanos ↔ terminología oficial de la Universidad Nacional del Valle Alto (UNVA)
INSTITUTIONAL_SYNONYMS = {
    "mensualidad": ["pago", "arancel", "cashea", "cuota"],
    "matricula": ["pago", "arancel", "inscripcion", "cashea"],
    "costo": ["pago", "arancel", "cashea"],
    "precio": ["pago", "arancel", "cashea"],
    "plata": ["pago", "arancel", "cashea"],
    "raspar": ["inasistencia", "pi", "nota", "reprobar"],
    "reprobar": ["inasistencia", "pi", "nota"],
    "falta": ["inasistencia", "pi"],
    "tesis": ["grado", "titulo", "pasantia"],
    "graduacion": ["grado", "titulo", "solvencia"],
    "horario": ["turno", "sabatino", "diurno", "nocturno"],
    "calificacion": ["nota", "evaluacion", "escala"],
    "boletin": ["nota", "promedio", "ternanet"],
}


def expand_query_synonyms(tokens: List[str]) -> List[str]:
    """Expande tokens de consulta con sinónimos institucionales de la Universidad Nacional del Valle Alto (UNVA)."""
    expanded = list(tokens)
    seen = set(tokens)
    for t in tokens:
        for syn in INSTITUTIONAL_SYNONYMS.get(t, []):
            if syn not in seen:
                expanded.append(syn)
                seen.add(syn)
    return expanded


def _stem_spanish_plural(word: str) -> str:
    """Normalización ligera de plurales en español sin dependencias externas."""
    if len(word) <= 4:
        return word
    if word.endswith("iones") and len(word) > 6:
        return word[:-2]
    if word.endswith(("ades", "ales", "ares", "ores", "enes", "eles")) and len(word) > 5:
        return word[:-2]
    if word.endswith("s") and not word.endswith(("is", "ss")):
        return word[:-1]
    return word


def tokenize_bm25(text: str) -> List[str]:
    """Tokeniza texto normalizado eliminando puntuación, stopwords y normalizando plurales."""
    norm = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", norm)
    # Conservar siglas académicas clave de 2 letras como 'pi' (Pérdida por Inasistencia)
    result = []
    for t in tokens:
        if (len(t) > 2 or t in {"pi"}) and t not in SPANISH_STOPWORDS:
            result.append(_stem_spanish_plural(t))
    return result


class RAGEngine:
    def __init__(self):
        self._cached_chunks: Optional[List[Dict[str, Any]]] = None
        self._bm25_df: Dict[str, int] = {}
        self._bm25_avgdl: float = 1.0
        self.session_memories: Dict[str, List[Dict[str, str]]] = {}
        self.session_students: Dict[str, str] = {}  # session_id -> cedula
        self._max_sessions = 500
        self._query_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl_seconds = 3600  # 1 hora de validez
        self._max_cache_entries = 250

    def _make_cache_key(self, query: str, student: Optional[Dict[str, Any]]) -> str:
        clean = re.sub(r"[^a-z0-9]", "", normalize_text(query.strip()))
        ced = student.get("cedula", "anon") if student else "anon"
        return f"{clean}::{ced}"

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if cache_key in self._query_cache:
            entry = self._query_cache[cache_key]
            if time.time() - entry["timestamp"] < self._cache_ttl_seconds:
                return entry["data"]
            else:
                del self._query_cache[cache_key]
        return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]):
        if len(self._query_cache) >= self._max_cache_entries:
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        self._query_cache[cache_key] = {
            "timestamp": time.time(),
            "data": data
        }

    def set_session_student(self, session_id: str, cedula: str):
        """Asocia una cédula de estudiante de TernaNet a una sesión (con límite LRU para evitar memory leaks)."""
        if session_id and cedula:
            if session_id not in self.session_students and len(self.session_students) >= self._max_sessions:
                oldest = next(iter(self.session_students))
                del self.session_students[oldest]
            clean = normalize_cedula(cedula)
            self.session_students[session_id] = clean

    def get_session_student(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene los datos de TernaNet del estudiante vinculado a la sesión."""
        if not session_id or session_id not in self.session_students:
            return None
        return get_student_by_cedula(self.session_students[session_id])

    def clear_session_student(self, session_id: str):
        """Desvincula el expediente estudiantil de una sesión para volver al modo general."""
        if session_id and session_id in self.session_students:
            del self.session_students[session_id]

    def clear_session(self, session_id: str):
        if session_id in self.session_memories:
            del self.session_memories[session_id]
        if session_id in self.session_students:
            del self.session_students[session_id]

    def _is_conversational_query(self, query: str) -> bool:
        """Detecta saludos, agradecimientos o preguntas sobre las capacidades del asistente inicial."""
        norm = normalize_text(query).strip()
        clean = re.sub(r"[^a-z0-9\s]", "", norm).strip()
        if not clean:
            return False
        greetings = {
            "hola", "buenas", "buenos dias", "buenas tardes", "buenas noches",
            "saludos", "hey", "holi", "que tal", "como estas", "como esta",
            "gracias", "muchas gracias", "ok", "okay", "perfecto", "genial",
            "ayuda", "menu", "inicio", "start"
        }
        if clean in greetings:
            return True
        patterns = [
            r"^(hola|buenas|buenos dias|buenas tardes|buenas noches|saludos)\b",
            r"\b(quien eres|que eres|que puedes hacer|en que me puedes ayudar|como funcionas|que haces)\b",
            r"^(muchas )?gracias\b"
        ]
        return any(re.search(p, clean) for p in patterns)

    def _conversational_reply(self, query: str, active_student: Optional[Dict[str, Any]] = None) -> str:
        """Respuesta cordial para el modo inicial (público) o personalizado ante saludos y consultas de ayuda."""
        norm = normalize_text(query).strip()
        if "gracias" in norm:
            if active_student:
                return f"¡Con mucho gusto, **{active_student.get('nombre')}**! 😊 Si necesitas consultar algo más sobre tu expediente o los reglamentos de la Universidad Nacional del Valle Alto (UNVA), aquí estoy para ayudarte."
            return "¡Con mucho gusto! 😊 Si tienes otra consulta sobre admisión, sedes, carreras, pagos o reglamentos de la Universidad Nacional del Valle Alto (UNVA), escríbeme cuando quieras."

        if active_student:
            return (
                f"¡Hola, **{active_student.get('nombre')}**! 👋 Qué gusto saludarte.\n\n"
                f"Actualmente estás en **Modo Personalizado (TernaNet)** ({active_student.get('carrera')} · Sede {active_student.get('sede')}). "
                "Puedes preguntarme directamente sobre:\n"
                "• 📊 Tus materias en curso y promedio acumulado\n"
                "• 🤝 Si cumples los requisitos para **Servicio Comunitario**\n"
                "• 💼 Tu estatus de **Pasantías** y solvencias\n"
                "• 💳 El estado de tus cuotas con **Cashea**\n"
                "• 📋 Cualquier normativa o reglamento oficial de la Universidad Nacional del Valle Alto (UNVA)\n\n"
                "¿En qué te puedo ayudar hoy?"
            )

        return (
            "¡Hola! 👋 Soy el **Asistente Virtual Oficial de la Universidad Nacional del Valle Alto (UNVA)** (Modo General).\n\n"
            "Puedo orientarte de inmediato sobre:\n"
            "• 📋 **Nuevo Ingreso:** requisitos, recaudos e inscripción en TernaNet\n"
            "• 🏢 **Sedes y Contacto:** Caracas, Guatire y Barcelona / Puerto La Cruz\n"
            "• 🎓 **Oferta Académica:** carreras TSU y turnos (diurno, nocturno, sabatino)\n"
            "• 💳 **Pagos y Cashea:** aranceles y financiamiento en cuotas\n"
            "• 📊 **Reglamentos:** escala de notas, cortes, pérdida por inasistencia (PI), servicio comunitario y pasantías\n\n"
            "💡 *Si ya eres estudiante de la Universidad Nacional del Valle Alto (UNVA), puedes pulsar **Identificarme (TernaNet)** arriba o escribir tu cédula aquí para activar el modo personalizado con tu expediente.*"
        )

    def reload_documents(self) -> int:
        """Limpia la memoria caché y recarga todos los documentos PDF activos."""
        self._cached_chunks = None
        self._bm25_df = {}
        self._bm25_avgdl = 1.0
        self._query_cache.clear()
        chunks = self._get_memory_chunks()
        return len(chunks)

    def _get_memory_chunks(self) -> List[Dict[str, Any]]:
        """Carga los PDFs en memoria y construye el índice Okapi BM25 (< 30MB RAM)."""
        if self._cached_chunks is None:
            self._cached_chunks = []
            df: Dict[str, int] = {}
            total_len = 0

            try:
                from src.config import DOCUMENTS_DIR
                from src.ingestion import extract_text_from_pdf, split_text_into_chunks
            except ImportError:
                from config import DOCUMENTS_DIR
                from ingestion import extract_text_from_pdf, split_text_into_chunks

            for pdf in DOCUMENTS_DIR.glob("*.pdf"):
                pages = extract_text_from_pdf(pdf)
                for p in pages:
                    for c in split_text_into_chunks(p["text"], 120, 30):
                        tokens = tokenize_bm25(c)
                        tf = Counter(tokens)
                        doc_len = len(tokens)
                        total_len += doc_len
                        for term in tf.keys():
                            df[term] = df.get(term, 0) + 1

                        self._cached_chunks.append({
                            "content": c,
                            "source": pdf.name,
                            "page": p["page"],
                            "tf": tf,
                            "len": doc_len,
                        })

            self._bm25_df = df
            n_docs = len(self._cached_chunks)
            self._bm25_avgdl = (total_len / n_docs) if n_docs > 0 else 1.0

        return self._cached_chunks

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        history: Optional[List[Dict[str, str]]] = None,
        min_score: float = 0.35
    ) -> List[Dict[str, Any]]:
        """Motor de recuperación Okapi BM25 en Python puro con normalización de tildes y stopwords."""
        chunks = self._get_memory_chunks()
        if not chunks:
            return []

        query_tokens = expand_query_synonyms(tokenize_bm25(query))

        # Incorporar términos clave del historial reciente con menor peso para preguntas de seguimiento
        history_tokens: List[str] = []
        if history:
            for item in history[-2:]:
                history_tokens.extend([t for t in tokenize_bm25(item.get("content", "")) if len(t) > 4])

        if not query_tokens and not history_tokens:
            return []

        n_docs = len(chunks)
        avgdl = self._bm25_avgdl or 1.0
        k1 = 1.5
        b = 0.75

        def _bm25_score(doc: Dict[str, Any], tokens: List[str], weight: float = 1.0) -> float:
            score = 0.0
            tf_map = doc["tf"]
            dl = doc["len"] or 1
            for term in tokens:
                freq = tf_map.get(term, 0)
                if freq == 0:
                    continue
                df = self._bm25_df.get(term, 0)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                denom = freq + k1 * (1.0 - b + b * (dl / avgdl))
                score += weight * idf * ((freq * (k1 + 1.0)) / denom)
            return score

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for c in chunks:
            s = _bm25_score(c, query_tokens, weight=1.0)
            if history_tokens:
                s += _bm25_score(c, history_tokens[:8], weight=0.35)
            if s >= min_score:
                scored.append((s, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        return [
            {
                "content": c["content"],
                "source": c["source"],
                "page": c["page"],
                "score": round(score, 4),
                "distance": 0
            }
            for score, c in top
        ]

    # --- Helpers compartidos entre modo síncrono y streaming SSE ---

    def _resolve_student(
        self, query: str, session_id: Optional[str], student_cedula: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if student_cedula:
            if session_id:
                self.set_session_student(session_id, student_cedula)
            return get_student_by_cedula(student_cedula)

        if session_id:
            detected = detect_cedula_in_text(query)
            if detected:
                self.set_session_student(session_id, detected)
            return self.get_session_student(session_id)

        return None

    def _record_session_turn(self, session_id: Optional[str], query: str, answer: str):
        if not session_id:
            return
        if session_id not in self.session_memories:
            if len(self.session_memories) >= self._max_sessions:
                oldest = next(iter(self.session_memories))
                del self.session_memories[oldest]
            self.session_memories[session_id] = []
        self.session_memories[session_id].append({"role": "user", "content": query})
        self.session_memories[session_id].append({"role": "assistant", "content": answer})
        self.session_memories[session_id] = self.session_memories[session_id][-6:]

    def _build_prompts(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        active_student: Optional[Dict[str, Any]]
    ) -> Tuple[str, str, List[str]]:
        """Construye system_instruction, user_prompt y la lista de fuentes únicas."""
        if contexts:
            context_text = "\n\n".join([
                f"[Fuente: {c['source']} - Pág. {c['page']}]\n{c['content']}"
                for c in contexts
            ])
            unique_sources = list({f"{c['source']} (pág. {c['page']})" for c in contexts})
        else:
            context_text = "No se recuperaron reglamentos específicos para esta pregunta; responde utilizando el expediente del estudiante si aplica."
            unique_sources = []

        history_text = ""
        if history:
            formatted_turns = []
            for h in history[-4:]:
                role_label = "Estudiante" if h["role"] == "user" else "Asistente"
                formatted_turns.append(f"{role_label}: {h['content']}")
            history_text = "HISTORIAL RECIENTE DE LA CONVERSACIÓN:\n" + "\n".join(formatted_turns) + "\n\n"

        system_instruction = (
            "Eres el Asistente Virtual Oficial del Universidad Nacional del Valle Alto (UNVA). "
            "Tu misión es orientar con amabilidad, claridad y precisión a la comunidad estudiantil. "
            "REGLAS OBLIGATORIAS:\n"
            "1. Sé conciso, directo y estructurado (usa viñetas breves; evita introducciones largas o rellenos innecesarios).\n"
            "2. Basa tu respuesta ESTRICTAMENTE en la información institucional proporcionada en el CONTEXTO.\n"
            "3. Recuerda y utiliza el HISTORIAL DE LA CONVERSACIÓN para responder preguntas de seguimiento (ej. pronombres como 'allí', 'eso', 'esa sede', etc.).\n"
            "4. Menciona el nombre del reglamento, guía o artículo de forma natural (ej. 'Según el Reglamento de Evaluación Estudiantil...'), pero NO repitas nombres técnicos de archivos '.pdf' ni escribas '(Fuente: ...)' al final de cada viñeta, ya que el sistema muestra las fuentes oficiales automáticamente al pie del mensaje.\n"
            "5. Si el contexto no contiene la respuesta completa, indica cordialmente que el estudiante debe acudir a la taquilla de Control de Estudios o consultar en TernaNet.\n"
            "6. Responde siempre en español formal pero cercano."
        )

        student_prompt_inject = ""
        if active_student:
            student_context = format_student_context_for_rag(active_student)
            student_prompt_inject = f"{student_context}\n\n"
            system_instruction += (
                "\n\n=======================================================\n"
                "ATENCIÓN - PERFIL DEL ESTUDIANTE IDENTIFICADO EN TERNANET:\n"
                f"{student_context}\n"
                "DIRECTRICES OBLIGATORIAS PARA ESTE ESTUDIANTE:\n"
                f"1. Dirígete cordialmente al estudiante por su nombre ('{active_student.get('nombre')}').\n"
                "2. Cruza los requisitos y normativas de la Universidad Nacional del Valle Alto (UNVA) (créditos exigidos, materias cursadas, semestres, prelaciones, servicio comunitario, pasantías, cuotas de Cashea y solvencias) DIRECTAMENTE con los datos reales de su expediente.\n"
                "   - Ejemplo Servicio Comunitario: En la Universidad Nacional del Valle Alto (UNVA) se exige tener aprobado al menos el 50% de los créditos de la carrera. Verifica su porcentaje aprobado en el expediente e indícale con precisión si ya puede inscribirlo o cuántos créditos le faltan.\n"
                "   - Ejemplo Pasantías: Revisa si cumple con el avance de semestres/créditos y su estatus de pasantías.\n"
                "   - Ejemplo Pagos/Cashea: Si consulta sobre pagos o matrícula, dile con precisión sus cuotas pagadas/pendientes y fechas.\n"
                "3. Ofrece una orientación personalizada, motivadora y clara.\n"
                "=======================================================\n"
            )

        user_prompt = f"{history_text}{student_prompt_inject}CONTEXTO OFICIAL DEL UNVA:\n{context_text}\n\nNUEVA PREGUNTA DEL ESTUDIANTE: {query}"
        return system_instruction, user_prompt, unique_sources

    def generate_response(self, query: str, session_id: str = None, student_cedula: str = None) -> Dict[str, Any]:
        active_student = self._resolve_student(query, session_id, student_cedula)
        history = self.session_memories.get(session_id, []) if session_id else []

        # ⚡ Optimización 1: In-Memory Cache Lookup (< 10ms para consultas repetidas)
        cache_key = self._make_cache_key(query, active_student)
        cached_result = self._get_from_cache(cache_key)
        if cached_result and not history:
            self._record_session_turn(session_id, query, cached_result["answer"])
            return {
                **cached_result,
                "student": active_student,
                "cached": True
            }

        # 1. Recuperación BM25
        contexts = self.retrieve(query, history=history)

        # Saludos o preguntas de ayuda en modo inicial/personalizado (sin forzar login)
        if not contexts and self._is_conversational_query(query):
            msg = self._conversational_reply(query, active_student)
            self._record_session_turn(session_id, query, msg)
            return {
                "answer": msg,
                "sources": [],
                "student": active_student
            }

        # Solo retornar vacío si no hay contexto normativo NI perfil de estudiante activo
        if not contexts and not active_student:
            msg = "No encontré información relevante en los reglamentos y guías oficiales de la Universidad Nacional del Valle Alto (UNVA) para tu consulta. Te sugiero dirigirte a la Coordinación de Control de Estudios en la sede correspondiente."
            return {
                "answer": msg,
                "sources": [],
                "student": active_student
            }

        system_instruction, user_prompt, unique_sources = self._build_prompts(
            query, contexts, history, active_student
        )

        # 2. Strategy A: Google Gemini
        current_api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        if current_api_key:
            client = gemini_client
            if client is None and HAS_GEMINI:
                try:
                    from google import genai
                    client = genai.Client(api_key=current_api_key, http_options={"api_version": "v1beta"})
                except Exception as e:
                    print(f"[!] Error creando genai.Client: {e}")

            if client:
                from google.genai import types as genai_types
                gen_config = genai_types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=600,
                )
                model_name = os.getenv("GEMINI_MODEL") or GEMINI_MODEL or "gemini-3.1-flash-lite"
                models_to_try = list(dict.fromkeys([model_name, "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-flash-lite-latest"]))
                full_prompt = f"{system_instruction}\n\n{user_prompt}"
                for candidate_model in models_to_try:
                    try:
                        response = client.models.generate_content(
                            model=candidate_model,
                            contents=full_prompt,
                            config=gen_config
                        )
                        if response and response.text:
                            answer = response.text
                            self._record_session_turn(session_id, query, answer)
                            if not history:
                                self._save_to_cache(cache_key, {
                                    "answer": answer,
                                    "sources": unique_sources
                                })
                            return {
                                "answer": answer,
                                "sources": unique_sources,
                                "student": active_student
                            }
                    except Exception as e:
                        print(f"[!] Error con {candidate_model}: {e}. Probando siguiente modelo...")

        # 3. Strategy B: OpenAI
        if OPENAI_API_KEY and HAS_OPENAI:
            try:
                client = OpenAI(api_key=OPENAI_API_KEY)
                completion = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3
                )
                answer = completion.choices[0].message.content
                self._record_session_turn(session_id, query, answer)
                if not history:
                    self._save_to_cache(cache_key, {"answer": answer, "sources": unique_sources})
                return {
                    "answer": answer,
                    "sources": unique_sources,
                    "student": active_student
                }
            except Exception as e:
                print(f"[!] Error al invocar OpenAI API: {e}")

        # 4. Strategy C: High-Fidelity Local Synthesis (Fallback sin API key o ante fallo temporal de red)
        answer = self._synthesize_local(query, contexts, student=active_student)
        self._record_session_turn(session_id, query, answer)
        # Solo cachear síntesis local si NO hay API key configurada (evita cachear fallos temporales de Gemini)
        if not history and not current_api_key:
            self._save_to_cache(cache_key, {"answer": answer, "sources": unique_sources})
        return {
            "answer": answer,
            "sources": unique_sources,
            "student": active_student,
            "is_fallback": True
        }

    async def generate_response_stream(
        self, query: str, session_id: str = None, student_cedula: str = None
    ) -> AsyncGenerator[str, None]:
        """Transmite la respuesta token por token de forma asíncrona no bloqueante (SSE)."""
        active_student = self._resolve_student(query, session_id, student_cedula)

        # Enviar evento inicial de metadatos
        yield f"data: {json.dumps({'type': 'start', 'student': active_student})}\n\n"

        history = self.session_memories.get(session_id, []) if session_id else []
        cache_key = self._make_cache_key(query, active_student)
        cached_result = self._get_from_cache(cache_key)

        # ⚡ Si está en caché: entrega inmediata en un solo evento (< 10ms en servidor)
        if cached_result and not history:
            full_text = cached_result["answer"]
            yield f"data: {json.dumps({'type': 'token', 'content': full_text})}\n\n"

            self._record_session_turn(session_id, query, full_text)
            query_id = analytics.record_query(
                query=query,
                channel="web",
                answer=full_text,
                sources=cached_result.get("sources", [])
            )
            yield f"data: {json.dumps({'type': 'done', 'sources': cached_result.get('sources', []), 'student': active_student, 'cached': True, 'query_id': query_id})}\n\n"
            return

        contexts = self.retrieve(query, history=history)

        if not contexts and self._is_conversational_query(query):
            msg = self._conversational_reply(query, active_student)
            self._record_session_turn(session_id, query, msg)
            query_id = analytics.record_query(
                query=query,
                channel="web",
                answer=msg,
                sources=[],
                is_gap_override=False
            )
            yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': [], 'student': active_student, 'query_id': query_id})}\n\n"
            return

        if not contexts and not active_student:
            msg = "No encontré información relevante en los reglamentos y guías oficiales de la Universidad Nacional del Valle Alto (UNVA) para tu consulta. Te sugiero dirigirte a la Coordinación de Control de Estudios en la sede correspondiente."
            query_id = analytics.record_query(
                query=query,
                channel="web",
                answer=msg,
                sources=[],
                is_gap_override=True
            )
            yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': [], 'student': active_student, 'query_id': query_id})}\n\n"
            return

        system_instruction, user_prompt, unique_sources = self._build_prompts(
            query, contexts, history, active_student
        )
        full_prompt = f"{system_instruction}\n\n{user_prompt}"

        current_api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        stream_success = False
        full_answer = ""

        if current_api_key and gemini_client:
            from google.genai import types as genai_types
            gen_config = genai_types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=600,
            )
            model_name = os.getenv("GEMINI_MODEL") or GEMINI_MODEL or "gemini-3.1-flash-lite"
            models_to_try = list(dict.fromkeys([model_name, "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-flash-lite-latest"]))
            for candidate_model in models_to_try:
                try:
                    # Streaming asíncrono nativo (no bloquea el event loop de FastAPI)
                    stream = await gemini_client.aio.models.generate_content_stream(
                        model=candidate_model,
                        contents=full_prompt,
                        config=gen_config
                    )
                    async for chunk in stream:
                        if chunk.text:
                            full_answer += chunk.text
                            yield f"data: {json.dumps({'type': 'token', 'content': chunk.text})}\n\n"
                    if full_answer:
                        stream_success = True
                        break
                except Exception as e:
                    print(f"[!] Error streaming async con {candidate_model}: {e}")
                    # Si ya emitimos texto parcial al cliente, no reintentar otro modelo para evitar duplicar texto
                    if full_answer:
                        stream_success = True
                        break

        is_local_fallback = False
        if not stream_success and not full_answer:
            res = await asyncio.to_thread(
                self.generate_response, query, session_id=session_id, student_cedula=student_cedula
            )
            full_answer = res["answer"]
            unique_sources = res.get("sources", unique_sources)
            is_local_fallback = bool(res.get("is_fallback", False))
            words = full_answer.split(" ")
            for i in range(0, len(words), 5):
                chunk = " ".join(words[i:i + 5]) + " "
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                await asyncio.sleep(0.015)

        self._record_session_turn(session_id, query, full_answer)
        if not history and not (is_local_fallback and current_api_key):
            self._save_to_cache(cache_key, {
                "answer": full_answer,
                "sources": unique_sources
            })

        query_id = analytics.record_query(
            query=query,
            channel="web",
            answer=full_answer,
            sources=unique_sources,
            is_gap_override=False if active_student else None
        )

        yield f"data: {json.dumps({'type': 'done', 'sources': unique_sources, 'student': active_student, 'query_id': query_id})}\n\n"

    def _synthesize_local(self, query: str, contexts: List[Dict[str, Any]], student: Optional[Dict[str, Any]] = None) -> str:
        """Síntesis local inteligente cuando no se ha configurado una API Key de LLM."""
        if not contexts:
            if student:
                return (
                    f"¡Hola, **{student.get('nombre')}**! 🎓\n\n"
                    f"• **Carrera:** {student.get('carrera')} (Sede {student.get('sede')})\n"
                    f"• **Semestre actual:** {student.get('semestre')} ({student.get('porcentaje_carrera')}% de créditos aprobados)\n"
                    f"• **Promedio acumulado:** {student.get('promedio')} / 20\n"
                    f"• **Estatus arancelario:** {student.get('estatus_arancelario')}"
                )
            return "No encontré información relevante en los reglamentos oficiales de la Universidad Nacional del Valle Alto (UNVA) para tu consulta."

        key_terms = tokenize_bm25(query)
        best_doc = contexts[0]
        intro = f"De acuerdo con la documentación oficial de la **UNVA** (en *{best_doc['source']}*):\n\n"
        if student:
            intro = f"¡Hola, **{student.get('nombre')}**! 🎓 Revisando tu expediente en **TernaNet** ({student.get('carrera')}, {student.get('semestre')}do/to semestre) y la documentación oficial de la **UNVA**:\n\n"

        all_sentences = []
        seen_texts = set()

        for c in contexts:
            cleaned = re.sub(r'UNVA Oficial\s*-\s*C[oó]digo:[^\|]+\|\s*P[aá]gina\s*\d+/\d+', '', c["content"])
            cleaned = re.sub(r'UNIVERSIDAD NACIONAL DEL VALLE ALTO[^A-Z]+', '', cleaned)
            cleaned = re.sub(r'Direcci[oó]n Acad[eé]mica[^A-Z]+', '', cleaned)

            parts = re.split(r'(?:\. |\n+|(?=[1-9]\.\s)|(?=[•\-])|(?=Art[íi]culo\s+\d+))', cleaned)
            for part in parts:
                p = part.strip()
                if len(p) > 20 and not p.startswith("Página") and p.lower() not in seen_texts:
                    seen_texts.add(p.lower())
                    p_norm = normalize_text(p)
                    score = sum(2 for term in key_terms if term in p_norm)
                    all_sentences.append((score, p))

        all_sentences.sort(key=lambda x: x[0], reverse=True)

        relevant_sentences = [s[1] for s in all_sentences if s[0] > 0][:5]
        if not relevant_sentences:
            relevant_sentences = [s[1] for s in all_sentences[:4]]

        points = []
        for s in relevant_sentences:
            s_clean = s.lstrip("•- \t").rstrip(".,;")
            if s_clean:
                points.append(f"• **{s_clean}**." if len(s_clean) < 60 else f"• {s_clean}.")

        summary = "\n\n".join(points)
        return intro + summary

    def transcribe_audio(self, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        """Transcribe notas de voz estudiantiles usando Gemini Multimodal."""
        if not gemini_client:
            raise RuntimeError("Gemini Client no está configurado.")

        from google.genai import types

        m = (mime_type or "audio/ogg").lower()
        if "mp3" in m or "mpeg" in m:
            clean_mime = "audio/mp3"
        elif "wav" in m:
            clean_mime = "audio/wav"
        elif "m4a" in m or "aac" in m:
            clean_mime = "audio/aac"
        elif "flac" in m:
            clean_mime = "audio/flac"
        else:
            clean_mime = "audio/ogg"

        part = types.Part.from_bytes(data=audio_bytes, mime_type=clean_mime)
        prompt = (
            "Eres el transcriptor del Asistente Académico del Universidad Nacional del Valle Alto (UNVA) en Venezuela.\n"
            "Tu tarea es transcribir textualmente en español la nota de voz o consulta enviada por el estudiante.\n\n"
            "GLOSARIO OBLIGATORIO Y REGLAS DE CORRECCIÓN FONÉTICA:\n"
            "• 'Cashea': Método de pago en cuotas de la Universidad Nacional del Valle Alto (UNVA). Si el estudiante dice 'caché', 'cashe', 'cajea' o 'cash', DEBES transcribirlo como 'Cashea'.\n"
            "• 'TernaNet': Sistema y portal de la Universidad Nacional del Valle Alto (UNVA). Si suena 'internet', 'ternanet', 'tierranet', transcribe 'TernaNet'.\n"
            "• 'UNVA': Siglas del instituto.\n"
            "• 'PI': Pérdida de materia por inasistencias.\n"
            "• Sedes: Guatire, Caracas, Barcelona, Puerto La Cruz, Anzoátegui.\n"
            "• Carreras: Informática, Administración, Contaduría, Publicidad, Mercadeo.\n"
            "• Términos frecuentes: Servicio Comunitario, Pasantías, Solvencias, Pensum, Grado, Semestre, Materias preladas, Arancel.\n\n"
            "INSTRUCCIONES:\n"
            "1. Transcribe con total exactitud lo que dice el hablante, aplicando el glosario institucional para nombres propios y términos clave.\n"
            "2. Responde ÚNICAMENTE con la transcripción exacta. No agregues saludos, explicaciones, comillas ni texto introductorio."
        )
        try:
            config = types.GenerateContentConfig(temperature=0.0)
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[part, prompt],
                config=config
            )
            if response and response.text:
                return response.text.strip().strip('"').strip("'")
            return ""
        except Exception as e:
            print(f"[!] Error transcribiendo audio con Gemini ({GEMINI_MODEL}): {e}")
            raise


# Global instance
rag_engine = RAGEngine()

if __name__ == "__main__":
    test_q = "¿Cuáles son los requisitos de nuevo ingreso para bachilleres?"
    res = rag_engine.generate_response(test_q)
    print("Pregunta:", test_q)
    print("Respuesta:\n", res["answer"])
    print("Fuentes:", res["sources"])
