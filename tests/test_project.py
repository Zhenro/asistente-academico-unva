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
from src.api import app, ADMIN_PASSWORD
from src.rag_engine import rag_engine, normalize_text, tokenize_bm25


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

    def test_bm25_retrieves_accent_insensitive(self):
        results = rag_engine.retrieve("requisitos de inscripcion y cashea")
        self.assertTrue(len(results) > 0, "BM25 debe recuperar fragmentos relevantes sin tildes")
        self.assertGreater(results[0]["score"], 0.35)

    def test_bm25_detects_knowledge_gap_for_irrelevant_query(self):
        results = rag_engine.retrieve("receta de pasta carbonara con parmesano")
        self.assertEqual(results, [], "Consultas fuera de dominio no deben devolver chunks arbitrarios")


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


class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
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


if __name__ == "__main__":
    unittest.main()
