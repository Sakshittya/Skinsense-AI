"""Train a simple CNN baseline on class-folder image data.
Input layout: data/skin500/<class_name>/*.(jpg|png|webp)
For serious evaluation, split by patient/lesion, not by near-duplicate images.
"""
import argparse, json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',default='data/skin500'); p.add_argument('--epochs',type=int,default=8); p.add_argument('--batch-size',type=int,default=16); a=p.parse_args()
 data=Path(a.data)
 if not data.exists(): raise SystemExit(f'Dataset folder not found: {data.resolve()}')
 train=tf.keras.utils.image_dataset_from_directory(data, validation_split=.2, subset='training', seed=42, image_size=(224,224), batch_size=a.batch_size)
 val=tf.keras.utils.image_dataset_from_directory(data, validation_split=.2, subset='validation', seed=42, image_size=(224,224), batch_size=a.batch_size)
 classes=train.class_names
 if len(classes)<2: raise SystemExit('At least two labelled classes are required.')
 autotune=tf.data.AUTOTUNE; train=train.cache().shuffle(400).prefetch(autotune); val=val.cache().prefetch(autotune)
 model=models.Sequential([layers.Input((224,224,3)),layers.Rescaling(1./255),layers.RandomFlip('horizontal'),layers.RandomRotation(.08),layers.Conv2D(32,3,activation='relu'),layers.MaxPooling2D(),layers.Conv2D(64,3,activation='relu'),layers.MaxPooling2D(),layers.Conv2D(128,3,activation='relu'),layers.MaxPooling2D(),layers.GlobalAveragePooling2D(),layers.Dropout(.35),layers.Dense(64,activation='relu'),layers.Dense(len(classes),activation='softmax')])
 model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
 model.fit(train,validation_data=val,epochs=a.epochs,callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)])
 out=Path('models'); out.mkdir(exist_ok=True); model.save(out/'skin_cnn.keras'); (out/'class_names.json').write_text(json.dumps(classes,indent=2),encoding='utf-8')
 print('Saved models/skin_cnn.keras and class names:',classes)
 print('Do not treat validation accuracy on a tiny dataset as clinical performance.')
if __name__=='__main__': main()
