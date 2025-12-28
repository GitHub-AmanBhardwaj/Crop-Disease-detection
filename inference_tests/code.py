import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

MOBILE_MODEL_PATH = '/kaggle/input/mymodels/best_disease_model.keras'
EFFICIENT_MODEL_PATH = '/kaggle/input/mymodels/best_disease_model_b3.keras'
ITERATIONS = 100
WARMUP_ROUNDS = 10

def benchmark_model(model_path, input_shape, model_name="Model"):
    print(f"Loading {model_name} from {model_path}...")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

    dummy_input = np.random.random((1, *input_shape, 3)).astype(np.float32)

    print(f"--- Benchmarking {model_name} ---")
    print(f"Input Shape: {dummy_input.shape}")
    
    print("Warming up...", end=" ")
    for _ in range(WARMUP_ROUNDS):
        _ = model.predict(dummy_input, verbose=0)
    print("Done.")

    latencies = []
    print(f"Running {ITERATIONS} iterations...")
    
    for i in range(ITERATIONS):
        start_time = time.perf_counter()
        _ = model.predict(dummy_input, verbose=0)
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000)

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    print(f"\nResults for {model_name}:")
    print(f"  Average Latency: {avg_latency:.2f} ms")
    print(f"  Std Deviation:   {std_latency:.2f} ms")
    print(f"  Min / Max:       {min_latency:.2f} ms / {max_latency:.2f} ms")
    print("-" * 40)
    
    return avg_latency, std_latency

if __name__ == "__main__":
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    mn_avg, mn_std = benchmark_model(MOBILE_MODEL_PATH, (224, 224), "MobileNetV2")

    en_avg, en_std = benchmark_model(EFFICIENT_MODEL_PATH, (300, 300), "EfficientNetB3")

    if mn_avg and en_avg:
        speedup = en_avg / mn_avg
        print(f"\nSUMMARY:")
        print(f"MobileNetV2 is {speedup:.2f}x faster than EfficientNetB3.")
