import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

image_size = 224
batch_size = 32
warmup_epochs = 5
finetune_epochs = 10

train_dir = 'final_dataset/train'
val_dir = 'final_dataset/validation'
test_dir = 'final_dataset/test'

# 1. Enhanced Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# 2. Build MobileNetV2 architecture with Dropout
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
base_model.trainable = False  # Freeze base model for initial warmup stage

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)  # Regularization to prevent overfitting
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# 3. Callbacks setup
checkpoint = ModelCheckpoint('healthy_vs_rotten.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

callbacks = [checkpoint, reduce_lr, early_stop]

# --- STAGE 1: Warmup Training ---
print("\n" + "="*50)
print("🚀 STAGE 1: Training Classification Head (Warmup)")
print("="*50)

model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=warmup_epochs, callbacks=callbacks)

# --- STAGE 2: Fine-Tuning Top Layers ---
print("\n" + "="*50)
print("🔥 STAGE 2: Unfreezing Top 30 MobileNetV2 Layers (Fine-Tuning)")
print("="*50)

base_model.trainable = True
# Freeze all layers except the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with smaller learning rate to prevent destroying pre-trained feature weights
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, validation_data=val_generator, epochs=finetune_epochs, callbacks=callbacks)

print("\n✅ Upgraded training complete! The new best model is saved to healthy_vs_rotten.h5.")
