#!/usr/bin/env python3
import sys

import axengine as axe


def main(model_path):
    print(f"[INFO] Testing AxEngine with model: {model_path}")

    available = axe.get_available_providers()
    print(f"[INFO] Available providers: {available}")

    if axe.axengine_provider_name not in available:
        print(f"[ERROR] {axe.axengine_provider_name} not available")
        return False

    try:
        session = axe.InferenceSession(model_path, providers=[axe.axengine_provider_name])
        print(f"[INFO] Successfully created session with {axe.axengine_provider_name}")

        inputs = session.get_inputs()
        outputs = session.get_outputs()
        print(f"[INFO] Model inputs: {len(inputs)}, outputs: {len(outputs)}")

        return True
    except Exception as e:
        print(f"[ERROR] Failed to create session: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_axengine_basic.py <model_path>")
        sys.exit(1)

    success = main(sys.argv[1])
    sys.exit(0 if success else 1)
