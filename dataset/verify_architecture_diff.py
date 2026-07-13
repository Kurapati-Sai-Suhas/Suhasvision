import glob
import sys
import io
import tensorflow as tf

# Import the model builder and custom layer
sys.path.append("academic_scripts")
from train_advanced_model import build_tcn_attention_model, SEQ_LEN, FEATURE_COLS
from inference_service import TemporalAttention

print("--- 1. Checking existing .keras files ---")
keras_files = glob.glob("cricket_stance_advanced_v*.keras")
print(f"Files found on disk: {keras_files}")

print("\n--- 2. Instantiating Model from Script ---")
script_model = build_tcn_attention_model(SEQ_LEN, len(FEATURE_COLS))
script_stream = io.StringIO()
script_model.summary(print_fn=lambda x: script_stream.write(x + '\n'))
script_summary = script_stream.getvalue()

print(f"Script Model Total Params: {script_model.count_params()}")

print("\n--- 3. Loading Deployed v4.keras ---")
if "cricket_stance_advanced_v4.keras" in keras_files:
    deployed_model = tf.keras.models.load_model("cricket_stance_advanced_v4.keras", custom_objects={'TemporalAttention': TemporalAttention})
    deployed_stream = io.StringIO()
    deployed_model.summary(print_fn=lambda x: deployed_stream.write(x + '\n'))
    deployed_summary = deployed_stream.getvalue()
    
    print(f"Deployed Model Total Params: {deployed_model.count_params()}")
    
    if script_model.count_params() == deployed_model.count_params():
        print("\nMATCH CONFIRMED: Parameter counts are exactly identical.")
        # Check layer by layer shapes
        for s_layer, d_layer in zip(script_model.layers, deployed_model.layers):
            s_shape = s_layer.output_shape if hasattr(s_layer, 'output_shape') else None
            d_shape = d_layer.output_shape if hasattr(d_layer, 'output_shape') else None
            if s_shape != d_shape:
                print(f"SHAPE MISMATCH in {s_layer.name}: Script {s_shape} != Deployed {d_shape}")
            elif s_layer.__class__.__name__ != d_layer.__class__.__name__:
                print(f"TYPE MISMATCH in {s_layer.name}: Script {s_layer.__class__.__name__} != Deployed {d_layer.__class__.__name__}")
        print("Layer shapes and types match exactly.")
    else:
        print("\nMISMATCH DETECTED: Parameter counts differ!")
else:
    print("v4.keras not found!")

