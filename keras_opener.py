import tensorflow as tf

model = tf.keras.models.load_model("./models/best_tcn.keras")
model.summary()

# Parameter count and estimated size
total_params = model.count_params()
size_kb = (total_params * 4) / 1024
print(f"\nFloat32 size: {size_kb:.1f} KB")
print(f"INT8 size:    {size_kb/4:.1f} KB")