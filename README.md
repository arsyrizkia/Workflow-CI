# Workflow-CI

Kriteria 3 — Proyek Akhir kelas **Membangun Sistem Machine Learning** (Dicoding).

Workflow CI (GitHub Actions) melakukan re-training model Heart Disease via **MLflow Project** setiap kali trigger terpantik, mengunggah artefak mlruns, meng-commit model terlatih, lalu mem-build image dengan `mlflow models build-docker` dan push ke Docker Hub.
