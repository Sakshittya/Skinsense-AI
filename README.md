# SkinSense AI — Deep Learning Skin Screening & Care Navigator

A new project built from scratch. It borrows only the general UI/UX direction and useful workflow ideas from MediLink AI; it does not modify or reuse its source code.

## Stack
- Frontend: React + Vite, plain CSS (light, responsive, card-based design)
- Backend: Python + Flask
- Vision model: TensorFlow/Keras CNN classifier (training script included)
- Text model: TensorFlow/Keras LSTM next-token language model (training script included)
- Reports: downloadable PDF using ReportLab
- Doctor discovery: opens OpenStreetMap search results in a new tab; no maps API key required
- Storage: JSON demo history for immediate local use. MySQL is intentionally not wired in; you can add it later.

## Important medical limitation
This is an educational prototype, not a diagnostic device. It must not be used to diagnose, rule out cancer, recommend treatment, or delay medical care. Predictions are only as reliable as the dataset/model. The starter project intentionally does not fabricate accuracy: before training, the classifier returns a clear "model not trained" message. Seek a qualified dermatologist for diagnosis. Rapidly changing, bleeding, painful, infected, or concerning lesions should be assessed promptly by a clinician.

## Run locally (Windows PowerShell)
1. Install Python 3.10 or 3.11 and Node.js 20+.
2. Backend:
   ```powershell
   cd backend
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python app.py
   ```
3. Frontend (new terminal):
   ```powershell
   cd frontend
   npm install
   npm run dev
   ```
4. Open the Vite URL printed in terminal (normally http://localhost:5173).

## Dataset: 500 images
The project does not bundle medical images because dataset licensing and redistribution rights vary. Use the official ISIC/HAM10000 source and its terms:
- ISIC datasets and licenses: https://challenge.isic-archive.com/data/
- HAM10000 dataset paper: https://doi.org/10.1038/sdata.2018.161

For a guided download using the Kaggle CLI (requires a Kaggle account/API token and acceptance of the dataset terms), run:
```powershell
cd backend
pip install kaggle
python scripts/download_ham10000.py
python scripts/select_500_images.py --input data/raw/extracted --labels data/raw/labels.csv --output data/skin500 --per-class 72 --max-images 500
python scripts/train_cnn.py --data data/skin500 --epochs 8
```
The full HAM10000 archive is several GB; the script downloads the archive before selecting a balanced subset of up to 500 labelled images. The selector chooses up to 72 images per class and may copy fewer than 500 if classes have fewer examples. A 500-image dataset is a small educational sample and is not clinically adequate.

## Train the LSTM text model
```powershell
cd backend
python scripts/train_lstm.py --text data/report_corpus.txt --epochs 25
```
The model is an NLP coursework demonstration. When trained, the LSTM can add an experimental language draft to the result/PDF. The core precautions and next steps always come from controlled educational content; LSTM output is not trusted for medical facts or precautions.

## Features
- Image upload and CNN inference endpoint
- Disease/lesion information library with cautious wording
- Condition-specific educational report and PDF export
- LSTM text-generation training pipeline for syllabus demonstration
- Find a dermatologist by city/postcode via OpenStreetMap search link
- Searchable condition library
- Follow-up reminders stored locally in JSON for the demo
- Login/register demo UI (local-only prototype; not production authentication)
- User dashboard and admin/model-status view
- Image-analysis history and downloadable reports
- Responsive, light UI with a new teal-blue-violet palette

## MySQL integration later
The frontend/backend are structured so MySQL can be added later. For production, replace JSON demo persistence and demo-only authentication with a MySQL schema, hashed passwords, server-side sessions, ownership checks, CSRF protection, rate limits, file validation, and private file storage. Do not store sensitive health images in public folders.
