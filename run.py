"""
Punto de entrada principal para el Asistente Virtual IUTECP.
Ejecuta:
    python run.py
"""
import os
import sys
import uvicorn
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import DOCUMENTS_DIR

def main():
    print("=" * 65)
    print("  IUTECP - ASISTENTE VIRTUAL DE CONSULTAS ACADÉMICAS (RAG)  ")
    print("=" * 65)
    
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("[!] No se detectaron PDFs en data/documents/. Ejecuta: python scripts/generate_documents.py")
    else:
        print(f"[+] {len(pdf_files)} documentos oficiales detectados en: {DOCUMENTS_DIR}")

    print("\n[+] Iniciando servidor web en: http://127.0.0.1:8000")
    print("[+] Abre tu navegador para interactuar con el Asistente Virtual")
    print("[+] Presiona CTRL + C para detener el servidor.\n")
    
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
