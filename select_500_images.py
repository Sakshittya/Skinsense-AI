"""Select up to 500 labelled images into class folders from a CSV manifest.
Expected labels.csv columns: filename,label. Images may be in nested folders under input.
"""
import argparse, random, shutil
from pathlib import Path
import pandas as pd

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--labels',required=True); p.add_argument('--output',required=True); p.add_argument('--per-class',type=int,default=72); p.add_argument('--max-images',type=int,default=500); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
 src=Path(a.input); out=Path(a.output); df=pd.read_csv(a.labels)
 if not {'filename','label'}.issubset(df.columns): raise SystemExit('CSV must contain filename,label columns.')
 files={f.name:f for f in src.rglob('*') if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}}
 df=df[df.filename.astype(str).isin(files)]
 if df.empty: raise SystemExit('No labelled images matched filenames. Check the manifest and input path.')
 random.seed(a.seed); copied=0
 for label, group in df.groupby('label'):
  rows=group.to_dict('records'); random.shuffle(rows); rows=rows[:a.per_class]
  dest=out/str(label); dest.mkdir(parents=True,exist_ok=True)
  for row in rows:
   if copied >= a.max_images: break
   srcfile=files[str(row['filename'])]; target=dest/srcfile.name
   if not target.exists(): shutil.copy2(srcfile,target); copied+=1
  if copied >= a.max_images: break
 print(f'Copied {copied} images to {out} (maximum requested: {a.max_images}). Review class balance and patient-level leakage before training.')
if __name__=='__main__': main()
