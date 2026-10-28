from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image, UnidentifiedImageError
import io, json, uuid, os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
UPLOADS = DATA / 'uploads'
DATA.mkdir(exist_ok=True); UPLOADS.mkdir(exist_ok=True)
HISTORY_FILE = DATA / 'history.json'
REMINDERS_FILE = DATA / 'reminders.json'
app = Flask(__name__)
CORS(app, resources={r'/api/*': {'origins': ['http://localhost:5173', 'http://127.0.0.1:5173']}})
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

# Educational reference text only. This is not individualized medical advice.
CONDITIONS = {
 'akiec': {'name':'Actinic keratoses (AKIEC)', 'summary':'A rough or scaly lesion that may be associated with cumulative sun exposure.', 'precautions':['Protect skin from sun with shade, clothing and broad-spectrum sunscreen.','Avoid picking or attempting to remove the lesion yourself.','Arrange assessment with a qualified clinician, especially if it changes, bleeds or does not heal.'], 'care':'A clinician can examine the lesion and determine whether further testing or treatment is needed.'},
 'bcc': {'name':'Basal cell carcinoma (BCC)', 'summary':'A skin lesion category associated with basal cell carcinoma in the training label set.', 'precautions':['Arrange prompt evaluation by a dermatologist or qualified clinician.','Use sun protection and avoid tanning beds.','Do not use an image prediction to rule out or confirm cancer.'], 'care':'Only a clinician can assess the lesion and determine appropriate tests and care.'},
 'bkl': {'name':'Benign keratosis-like lesion (BKL)', 'summary':'A broad label for benign keratosis-like appearances in the dataset; image classification alone cannot confirm benignity.', 'precautions':['Seek clinical review for a new or changing spot.','Avoid scraping, cutting or applying harsh products to the lesion.','Use sun protection and monitor changes in size, shape or color.'], 'care':'A clinician can distinguish look-alike lesions when needed.'},
 'df': {'name':'Dermatofibroma (DF)', 'summary':'A lesion category represented in the dataset; this result is not a confirmed diagnosis.', 'precautions':['Do not try to remove the spot at home.','Arrange clinical review if it changes, becomes painful or bleeds.','Seek advice if you are unsure what the lesion is.'], 'care':'A clinician can examine the spot and decide whether further assessment is needed.'},
 'mel': {'name':'Melanoma (MEL)', 'summary':'A serious skin-lesion category in the dataset. A model output cannot diagnose melanoma, but a concerning lesion warrants prompt clinical assessment.', 'precautions':['Please seek prompt assessment from a qualified dermatologist or clinician.','Do not wait for repeated app predictions if a lesion is changing or concerning.','Avoid treating or removing the lesion yourself.'], 'care':'Prompt professional assessment is important. This app cannot confirm or exclude melanoma.'},
 'nv': {'name':'Melanocytic nevus (NV)', 'summary':'A melanocytic nevus label in the dataset. The model cannot confirm that a particular spot is harmless.', 'precautions':['Arrange review if the spot is new, changing, asymmetrical, irregular, bleeding or itchy.','Use sun protection.','Do not rely on this prediction as proof that a lesion is benign.'], 'care':'A clinician can assess the lesion in the context of your skin history.'},
 'vasc': {'name':'Vascular lesion (VASC)', 'summary':'A vascular-lesion category in the dataset; the image result is only an educational model estimate.', 'precautions':['Seek clinical review for unexplained, changing or bleeding lesions.','Avoid puncturing or attempting to remove the lesion.','If bleeding is heavy or will not stop, seek urgent medical care.'], 'care':'A clinician can identify the lesion and recommend appropriate next steps.'}
}

def read_json(path, default):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError): return default

def write_json(path, data): path.write_text(json.dumps(data, indent=2), encoding='utf-8')

def model_status():
    return (BASE / 'models' / 'skin_cnn.keras').exists()

def generate_lstm_draft(seed_text, max_tokens=24):
    """Optional experimental prose draft. Never used for precautions or care decisions."""
    model_path = BASE / 'models' / 'report_lstm.keras'
    vocab_path = BASE / 'models' / 'text_vocabulary.json'
    if not model_path.exists() or not vocab_path.exists(): return None
    try:
        import numpy as np
        import tensorflow as tf
        vocab = json.loads(vocab_path.read_text(encoding='utf-8'))
        lookup = {w:i for i,w in enumerate(vocab)}
        import re
        tokens = re.findall(r"[a-z]+|[0-9]+|[^\w\s]", seed_text.lower())
        ids = [lookup.get(t,1) for t in tokens][-24:]
        ids = [0] * (24-len(ids)) + ids
        model = tf.keras.models.load_model(model_path)
        generated=[]
        for _ in range(max_tokens):
            probs=model.predict(np.asarray([ids],dtype='int32'),verbose=0)[0]
            # Slightly favor common tokens; exclude padding/unknown IDs.
            probs[:2]=0
            total=float(probs.sum())
            if total<=0: break
            probs=probs/total
            idx=int(np.random.choice(len(probs),p=probs))
            word=vocab[idx]
            if word in {'.','!','?'} and len(generated)>8:
                generated.append(word); break
            generated.append(word)
            ids=(ids+[idx])[-24:]
        draft=' '.join(generated).replace(' .','.').replace(' ,',',').replace(' !','!').replace(' ?','?')
        return draft.strip() or None
    except Exception:
        return None

def predict_image(image_path):
    model_path = BASE / 'models' / 'skin_cnn.keras'
    labels_path = BASE / 'models' / 'class_names.json'
    if not model_path.exists() or not labels_path.exists():
        return {'ready':False, 'message':'CNN model is not trained yet. Follow the dataset and training steps in README.md.'}
    try:
        import numpy as np
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        labels = json.loads(labels_path.read_text(encoding='utf-8'))
        img = Image.open(image_path).convert('RGB').resize((224,224))
        arr = np.asarray(img, dtype='float32') / 255.0
        probs = model.predict(arr[None, ...], verbose=0)[0]
        order = probs.argsort()[::-1][:3]
        top = []
        for i in order:
            key = labels[int(i)]
            item = CONDITIONS.get(key, {'name':key, 'summary':'No curated description available.', 'precautions':['Consult a qualified clinician for advice.'], 'care':'Professional assessment is recommended.'})
            top.append({'label':key, 'name':item['name'], 'confidence':round(float(probs[i])*100, 2)})
        return {'ready':True, 'top':top}
    except Exception as exc:
        return {'ready':False, 'message':f'Model inference failed: {type(exc).__name__}. Check the model and dependencies.'}

@app.get('/api/health')
def health(): return jsonify({'ok':True, 'service':'SkinSense AI', 'cnn_ready':model_status(), 'lstm_ready':(BASE/'models'/'report_lstm.keras').exists()})

@app.get('/api/conditions')
def conditions():
    q = request.args.get('q','').strip().lower()
    rows = [{'id':k, **v} for k,v in CONDITIONS.items()]
    if q: rows = [r for r in rows if q in r['name'].lower() or q in r['summary'].lower()]
    return jsonify(rows)

@app.post('/api/analyze')
def analyze():
    if 'image' not in request.files: return jsonify({'error':'Please attach an image file.'}), 400
    f = request.files['image']
    if not f.filename: return jsonify({'error':'Please choose an image.'}), 400
    try:
        img = Image.open(f.stream)
        if img.format not in ('JPEG','PNG','WEBP'): return jsonify({'error':'Use a JPG, PNG or WEBP image.'}), 400
        img.verify()
        f.stream.seek(0)
    except (UnidentifiedImageError, OSError): return jsonify({'error':'The selected file is not a valid image.'}), 400
    scan_id = uuid.uuid4().hex[:12]
    ext = '.png' if f.mimetype == 'image/png' else '.webp' if f.mimetype == 'image/webp' else '.jpg'
    path = UPLOADS / f'{scan_id}{ext}'
    f.save(path)
    prediction = predict_image(path)
    if not prediction.get('ready'):
        path.unlink(missing_ok=True)
        return jsonify({'ready':False, 'message':prediction.get('message'), 'next_steps':['Download an appropriately licensed labelled dataset.','Prepare up to 500 images using scripts/select_500_images.py.','Train the CNN using scripts/train_cnn.py.']}), 503
    top = prediction['top'][0]
    key = top['label']; condition = CONDITIONS.get(key, {'name':top['name'],'summary':'Educational estimate only.','precautions':['Consult a qualified clinician.'],'care':'Professional assessment recommended.'})
    result = {'id':scan_id,'created_at':datetime.now().isoformat(timespec='seconds'),'prediction':top,'alternatives':prediction['top'][1:], 'summary':condition['summary'], 'precautions':condition['precautions'], 'care':condition['care'], 'image_name':f.filename, 'lstm_draft':generate_lstm_draft('educational skin health report '+condition['summary']), 'disclaimer':'Educational screening only. Not a medical diagnosis. Consult a qualified clinician.'}
    history = read_json(HISTORY_FILE, [])
    history.insert(0, result); write_json(HISTORY_FILE, history[:100])
    return jsonify(result)

@app.get('/api/history')
def history(): return jsonify(read_json(HISTORY_FILE, []))

@app.post('/api/reminders')
def add_reminder():
    body = request.get_json(silent=True) or {}
    title = str(body.get('title','')).strip(); date = str(body.get('date','')).strip(); note = str(body.get('note','')).strip()
    if not title or not date: return jsonify({'error':'A reminder title and date are required.'}), 400
    rows = read_json(REMINDERS_FILE, [])
    item = {'id':uuid.uuid4().hex[:10], 'title':title[:100], 'date':date[:30], 'note':note[:500], 'done':False}
    rows.insert(0,item); write_json(REMINDERS_FILE, rows[:200]); return jsonify(item), 201

@app.get('/api/reminders')
def reminders(): return jsonify(read_json(REMINDERS_FILE, []))

@app.patch('/api/reminders/<rid>')
def update_reminder(rid):
    rows=read_json(REMINDERS_FILE, [])
    for r in rows:
        if r['id']==rid: r['done']=not r.get('done',False); write_json(REMINDERS_FILE,rows); return jsonify(r)
    return jsonify({'error':'Reminder not found'}),404

@app.post('/api/report/<rid>')
def report(rid):
    record = next((r for r in read_json(HISTORY_FILE, []) if r['id']==rid), None)
    if not record: return jsonify({'error':'Report record not found.'}),404
    output = io.BytesIO(); pdf=canvas.Canvas(output,pagesize=letter); w,h=letter; y=h-55
    pdf.setTitle('SkinSense AI Educational Report')
    pdf.setFont('Helvetica-Bold',18); pdf.drawString(48,y,'SkinSense AI'); y-=26
    pdf.setFont('Helvetica',10); pdf.drawString(48,y,'Educational skin-image screening report — not a diagnosis'); y-=28
    lines=[('Report ID',record['id']),('Date',record['created_at']),('Model estimate',record['prediction']['name']),('Model confidence',f"{record['prediction']['confidence']}%"),('Image file',record.get('image_name','Uploaded image')),('Summary',record.get('summary','')),('Experimental LSTM draft',record.get('lstm_draft') or 'Not generated: optional LSTM model has not been trained.'),('General next step',record.get('care',''))]
    for label,value in lines:
        pdf.setFont('Helvetica-Bold',10); pdf.drawString(48,y,label+':'); y-=14
        pdf.setFont('Helvetica',10)
        for line in wrap_text(str(value),95):
            pdf.drawString(58,y,line); y-=13
        y-=5
        if y<130: pdf.showPage(); y=h-55
    pdf.setFont('Helvetica-Bold',11); pdf.drawString(48,y,'General precautions'); y-=17; pdf.setFont('Helvetica',10)
    for p in record.get('precautions',[]):
        for line in wrap_text('• '+p,95): pdf.drawString(55,y,line); y-=13
        y-=4
    y-=10; pdf.setFont('Helvetica-Bold',10)
    for line in wrap_text('IMPORTANT: This report is generated for education only. AI can be wrong. It cannot confirm or rule out a skin condition or cancer. Please consult a qualified healthcare professional.',95): pdf.drawString(48,y,line); y-=13
    pdf.save(); output.seek(0)
    return send_file(output, mimetype='application/pdf', as_attachment=True, download_name=f'SkinSense-report-{rid}.pdf')

def wrap_text(text, width=95):
    words=text.split(); lines=[]; current=''
    for word in words:
        if len(current)+len(word)+1>width: lines.append(current); current=word
        else: current=(current+' '+word).strip()
    if current: lines.append(current)
    return lines

@app.get('/api/admin/summary')
def admin_summary():
    hist=read_json(HISTORY_FILE,[]); rem=read_json(REMINDERS_FILE,[])
    return jsonify({'scans':len(hist),'reminders':len(rem),'cnn_ready':model_status(),'lstm_ready':(BASE/'models'/'report_lstm.keras').exists(),'classes':len(CONDITIONS),'note':'Demo admin endpoint; add authenticated admin authorization before production.'})

if __name__=='__main__': app.run(host='127.0.0.1',port=5000,debug=True)
