import sys
import base64
import requests

API_URL = "http://localhost:8000/predict"

def test_predict(image_path):
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(API_URL, files=files)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return

    data = response.json()
    print(f"Predicted class: {data['predicted_class']}")
    print(f"Confidence: {data['confidence']*100:.1f}%")
    print("\nTop 3:")
    for item in data["top3"]:
        print(f"  {item['class_name']}: {item['confidence']*100:.1f}%")

    
    img_bytes = base64.b64decode(data["gradcam_image_base64"])
    output_path = "notebooks/api_test_gradcam.png"
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    print(f"\nSaved returned Grad-CAM image to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_api.py path/to/image.jpg")
        sys.exit(1)
    test_predict(sys.argv[1])
