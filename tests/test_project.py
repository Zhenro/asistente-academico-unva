"""
Suite de pruebas automatizadas para el Asistente Virtual Académico IUTECP.
Ejecutar con:
    python -m unittest discover -s tests -v
"""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import src.analytics as analytics
import src.ternanet as ternanet
import src.api as api_module
from src.api import app, ADMIN_PASSWORD
from src.rag_engine import RAGEngine, rag_engine, normalize_text, tokenize_bm25


class TestBM25AndRAG(unittest.TestCase):
    def test_normalize_text_removes_accents(self):
        self.assertEqual(normalize_text("Evaluación y Pasantías"), "evaluacion y pasantias")
        self.assertEqual(normalize_text("Inscripción IUTECP"), "inscripcion iutecp")

    def test_tokenize_filters_stopwords_and_keeps_pi(self):
        tokens = tokenize_bm25("¿Cómo funciona el PI en la evaluación?")
        self.assertIn("pi", tokens)
        self.assertIn("evaluacion", tokens)
        self.assertNotIn("como", tokens)
        self.assertNotIn("el", tokens)

    def test_spanish_plural_stemming(self):
        self.assertEqual(tokenize_bm25("pasantías"), tokenize_bm25("pasantía"))
        self.assertEqual(tokenize_bm25("créditos"), tokenize_bm25("crédito"))
        self.assertEqual(tokenize_bm25("inscripciones"), tokenize_bm25("inscripción"))

    def test_session_memory_is_bounded(self):
        engine = RAGEngine()
        engine._max_sessions = 3
        for i in range(5):
            engine.set_session_student(f"s_{i}", "30771246")
            engine._record_session_turn(f"s_{i}", f"q{i}", f"a{i}")
        self.assertEqual(len(engine.session_students), 3)
        self.assertEqual(len(engine.session_memories), 3)
        self.assertNotIn("s_0", engine.session_memories)

    def test_bm25_retrieves_accent_insensitive(self):
        results = rag_engine.retrieve("requisitos de inscripcion y cashea")
        self.assertTrue(len(results) > 0, "BM25 debe recuperar fragmentos relevantes sin tildes")
        self.assertGreater(results[0]["score"], 0.35)

    def test_bm25_detects_knowledge_gap_for_irrelevant_query(self):
        results = rag_engine.retrieve("receta de pasta carbonara con parmesano")
        self.assertEqual(results, [], "Consultas fuera de dominio no deben devolver chunks arbitrarios")

    def test_bm25_synonym_expansion(self):
        results = rag_engine.retrieve("como pago la mensualidad")
        self.assertTrue(len(results) > 0, "Sinónimo 'mensualidad' debe recuperar info de pagos/Cashea")

    def test_analytics_topic_with_accents_and_csv_safety(self):
        self.assertEqual(analytics._classify_topic("requisitos de inscripción"), "Admisión y TernaNet")
        self.assertEqual(analytics._classify_topic("ubicación sede Anzoátegui"), "Sedes y Contacto")
        self.assertEqual(analytics._classify_topic("carrera de Informática"), "Oferta Académica y Turnos")
        self.assertEqual(analytics._safe_csv_cell("=CMD|'/C calc'!A0"), "'=CMD|'/C calc'!A0")
        self.assertNotIn("\n", analytics._safe_csv_cell("linea 1\nlinea 2"))


class TestTernaNetStudents(unittest.TestCase):
    def test_all_seven_students_loaded(self):
        students = ternanet.load_students()
        cedulas = {s["cedula"] for s in students}
        expected = {
            "27346776", "28416131", "30566388",
            "30155465", "17498904", "30771246", "31418461"
        }
        self.assertTrue(expected.issubset(cedulas))

    def test_new_student_felix_ruiz(self):
        st = ternanet.get_student_by_cedula("30.771.246")
        self.assertIsNotNone(st)
        self.assertEqual(st["nombre"], "Félix Ruiz")
        self.assertEqual(ternanet.detect_cedula_in_text("Hola soy V-30.771.246"), "30771246")

    def test_new_student_paola_rada(self):
        st = ternanet.get_student_by_cedula("V-31.418.461")
        self.assertIsNotNone(st)
        self.assertEqual(st["nombre"], "Paola Rada")
        self.assertEqual(ternanet.detect_cedula_in_text("mi cédula es 31418461"), "31418461")
        ctx = ternanet.format_student_context_for_rag(st)
        self.assertIn("3er Semestre", ctx)


class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        api_module._RATE_LIMIT_BUCKETS.clear()
        api_module._ADMIN_LOGIN_BUCKETS.clear()
        self.client = TestClient(app)
        self._backup_analytics = analytics._load_data()

    def tearDown(self):
        analytics._save_data(self._backup_analytics)

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertGreater(data["memory_chunks"], 0)

    def test_students_list_includes_new_students(self):
        res = self.client.get("/api/students/list")
        self.assertEqual(res.status_code, 200)
        names = [s["nombre"] for s in res.json()]
        self.assertIn("Félix Ruiz", names)
        self.assertIn("Paola Rada", names)

    def test_identify_student_endpoint(self):
        res = self.client.post("/api/student/identify", json={
            "cedula": "31.418.461",
            "session_id": "test_paola_session"
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["student"]["nombre"], "Paola Rada")

    def test_reindex_requires_admin_key(self):
        res_unauth = self.client.post("/api/reindex")
        self.assertEqual(res_unauth.status_code, 401)

        res_auth = self.client.post("/api/reindex", headers={"x-admin-key": ADMIN_PASSWORD})
        self.assertEqual(res_auth.status_code, 200)
        self.assertEqual(res_auth.json()["status"], "success")

    def test_chat_stream_and_feedback_flow(self):
        # Forzar síntesis local para prueba determinista y rápida sin consumir cuota de API
        with patch("src.rag_engine.gemini_client", None):
            res = self.client.post("/api/chat/stream", json={
                "message": "¿Cuáles son las sedes del IUTECP?",
                "session_id": "test_stream_session"
            })
            self.assertEqual(res.status_code, 200)
            events = []
            for line in res.text.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

            done_events = [e for e in events if e.get("type") == "done"]
            self.assertEqual(len(done_events), 1)
            query_id = done_events[0].get("query_id")
            self.assertIsNotNone(query_id)

            fb = self.client.post("/api/feedback", json={
                "query_id": query_id,
                "rating": "up"
            })
            self.assertEqual(fb.status_code, 200)
            self.assertEqual(fb.json()["status"], "success")

    def test_upload_rejects_invalid_pdf_signature(self):
        res = self.client.post(
            "/api/admin/upload",
            headers={"x-admin-key": ADMIN_PASSWORD},
            files={"file": ("falso.pdf", b"esto no es un pdf real", "application/pdf")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("%PDF-", res.json()["detail"])

    def test_rate_limiting_triggers_429(self):
        with patch.object(api_module, "RATE_LIMIT_MAX_REQUESTS", 2):
            with patch("src.rag_engine.gemini_client", None):
                r1 = self.client.post("/api/chat", json={"message": "sedes del iutecp"})
                r2 = self.client.post("/api/chat", json={"message": "sedes del iutecp"})
                r3 = self.client.post("/api/chat", json={"message": "sedes del iutecp"})
                self.assertEqual(r1.status_code, 200)
                self.assertEqual(r2.status_code, 200)
                self.assertEqual(r3.status_code, 429)

    def test_general_mode_greeting_and_logout_flow(self):
        # 1. Modo General (sin estudiante logueado): saluda y orienta correctamente
        res_hi = self.client.post("/api/chat", json={
            "message": "hola",
            "session_id": "dual_mode_sess"
        })
        self.assertEqual(res_hi.status_code, 200)
        self.assertIsNone(res_hi.json()["student"])
        self.assertIn("Modo General", res_hi.json()["answer"])

        # 2. Activar Modo Personalizado con Paola Rada
        res_id = self.client.post("/api/student/identify", json={
            "cedula": "31.418.461",
            "session_id": "dual_mode_sess"
        })
        self.assertEqual(res_id.status_code, 200)

        # 3. Cerrar sesión de estudiante (volver a Modo General)
        res_out = self.client.post("/api/student/logout", json={
            "session_id": "dual_mode_sess"
        })
        self.assertEqual(res_out.status_code, 200)

        # 4. Siguiente consulta NO debe re-loguear a Paola Rada
        with patch("src.rag_engine.gemini_client", None):
            res_after = self.client.post("/api/chat", json={
                "message": "hola",
                "session_id": "dual_mode_sess"
            })
        self.assertEqual(res_after.status_code, 200)
        self.assertIsNone(res_after.json()["student"])
        self.assertNotIn("Paola", res_after.json()["answer"])

    def test_admin_login_rate_limit(self):
        with patch.object(api_module, "ADMIN_LOGIN_MAX_ATTEMPTS", 2):
            r1 = self.client.post("/api/admin/login", json={"password": "wrong"})
            r2 = self.client.post("/api/admin/login", json={"password": "wrong"})
            r3 = self.client.post("/api/admin/login", json={"password": "wrong"})
            self.assertEqual(r1.status_code, 401)
            self.assertEqual(r2.status_code, 401)
            self.assertEqual(r3.status_code, 429)

    def test_greeting_is_not_flagged_as_knowledge_gap(self):
        res = self.client.post("/api/chat/stream", json={
            "message": "hola",
            "session_id": "gap_check_sess"
        })
        self.assertEqual(res.status_code, 200)
        summary = analytics.get_analytics_summary()
        latest = summary["recent_queries"][0]
        self.assertEqual(latest["query"], "hola")
        self.assertFalse(latest["is_gap"])


if __name__ == "__main__":
    unittest.main()
