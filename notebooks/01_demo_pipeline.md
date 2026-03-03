# Demo pipeline (no private data)

```bash
python -m src.make_synthetic
python -m src.eda_summary


Commit.

---

## ✅ Prova che tutto funziona (1 test rapido)
Sul tuo PC (se hai Python), nella cartella repo:

```bash
pip install -r requirements.txt
python -m src.make_synthetic
python -m src.eda_summary
