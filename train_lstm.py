"""Train an LSTM next-token model for a report-language coursework demo.
Generated prose must never be used as medical advice; the API keeps all care guidance curated.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def tokenize(s): return re.findall(r"[a-z]+|[0-9]+|[^\w\s]", s.lower())
def main():
 p=argparse.ArgumentParser(); p.add_argument('--text',default='data/report_corpus.txt'); p.add_argument('--epochs',type=int,default=25); a=p.parse_args()
 path=Path(a.text)
 if not path.exists(): raise SystemExit(f'Text corpus not found: {path.resolve()}')
 tokens=tokenize(path.read_text(encoding='utf-8'))
 counts={}
 for t in tokens: counts[t]=counts.get(t,0)+1
 vocab=['<pad>','<unk>']+[t for t,_ in sorted(counts.items(),key=lambda x:(-x[1],x[0])) if t not in {'<pad>','<unk>'}][:2998]
 word_to_id={w:i for i,w in enumerate(vocab)}; ids=[word_to_id.get(t,1) for t in tokens]; seq=24
 xs=[]; ys=[]
 for i in range(seq,len(ids)): xs.append(ids[i-seq:i]); ys.append(ids[i])
 if len(xs)<30: raise SystemExit('Corpus too small. Add more educational sample text to report_corpus.txt.')
 x=np.asarray(xs,dtype='int32'); y=np.asarray(ys,dtype='int32'); n=len(vocab)
 model=models.Sequential([layers.Input((seq,)),layers.Embedding(n,64,mask_zero=True),layers.LSTM(96),layers.Dropout(.2),layers.Dense(n,activation='softmax')])
 model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']); model.fit(x,y,epochs=a.epochs,batch_size=32,validation_split=.1,callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)])
 out=Path('models'); out.mkdir(exist_ok=True); model.save(out/'report_lstm.keras'); (out/'text_vocabulary.json').write_text(json.dumps(vocab,indent=2),encoding='utf-8')
 print('Saved LSTM demo model and vocabulary. Generated text is experimental and must not supply medical facts or precautions.')
if __name__=='__main__': main()
