import glob
import sys
import io
import tensorflow as tf
from inference_service import TemporalAttention

keras_files = glob.glob("cricket_stance_advanced_v*.keras")
if not keras_files:
    print("No .keras files found!")
    sys.exit(0)

with open("model_summaries.txt", "w", encoding="utf-8") as f_out:
    for path in sorted(keras_files):
        f_out.write(f"=== Model: {path} ===\n")
        try:
            model = tf.keras.models.load_model(path, custom_objects={'TemporalAttention': TemporalAttention})
            
            # Capture model.summary() output
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            summary_string = stream.getvalue()
            stream.close()
            
            f_out.write(summary_string)
            f_out.write("\nLayer Details:\n")
            for layer in model.layers:
                output_shape = layer.output_shape if hasattr(layer, 'output_shape') else None
                f_out.write(f"{layer.name} | {layer.__class__.__name__} | Output: {output_shape}\n")
            
            f_out.write("\n" + "="*50 + "\n\n")
        except Exception as e:
            f_out.write(f"Failed to load {path}: {e}\n\n")

print("Summaries written to model_summaries.txt")
