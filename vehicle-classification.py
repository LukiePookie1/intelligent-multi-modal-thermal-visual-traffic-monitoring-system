import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import json
from datetime import datetime
import shutil
import sys
import tqdm
import subprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
resnet_model.to(device)
resnet_model.eval()

def upscale_and_enhance(image):
    desired_size = (256, 256)
    upscaled_image = image.resize(desired_size, Image.LANCZOS)
    return upscaled_image

preprocess_classification = transforms.Compose([
    transforms.Lambda(lambda img: upscale_and_enhance(img)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image_classification(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess_classification(img)
    return img_t

def classify_image(image_tensor, model, imagenet_labels):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top1_prob, top1_catid = torch.topk(probabilities, 1)

        if top1_prob[0].item() >= 0.4:
            class_label = imagenet_labels[top1_catid[0].item()]
            confidence_score = top1_prob[0].item()
            return class_label, confidence_score
    return None, None

def process_images(image_folder, imagenet_labels):
    image_count = 0
    classification_counts = {}
    image_classifications = {}
    classification_images = []

    start_time = datetime.now()

    image_files = [filename for filename in os.listdir(image_folder) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)

    progress_bar = tqdm.tqdm(total=total_images, unit="image", dynamic_ncols=True)

    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        image_tensor = preprocess_image_classification(image_path)
        class_label, confidence_score = classify_image(image_tensor, resnet_model, imagenet_labels)

        if class_label is not None:
            classification_counts[class_label] = classification_counts.get(class_label, 0) + 1
            image_classifications[filename] = (class_label, confidence_score)
            classification_images.append((filename, class_label, confidence_score))

        image_count += 1
        progress_bar.update(1)

    progress_bar.close()

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    return image_count, classification_counts, image_classifications, classification_images, processing_time

def generate_report(input_folder, image_count, processing_time, classification_counts, image_classifications, classification_images):
    minutes, seconds = divmod(int(processing_time), 60)
    classification_report_lines = [
        "Image Classification Report",
        "===========================",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Input Image Folder: {os.path.basename(input_folder)}",
        "",
        f"Total Images Analyzed: {image_count}",
        f"Processing Time: {minutes:02d}:{seconds:02d}",
        "",
        "Classification Counts:",
    ]

    for class_label, count in sorted(classification_counts.items()):
        classification_report_lines.append(f"{class_label}: {count}")

    classification_report_lines.append("")
    classification_report_lines.append("Image Classifications:")

    for image_name in sorted(os.listdir(input_folder)):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            if image_name in image_classifications:
                class_label, confidence_score = image_classifications[image_name]
                classification_report_lines.append(f"{image_name}: {class_label} ({confidence_score:.2f})")
            else:
                classification_report_lines.append(f"{image_name}: Not classified")

    classification_report_content = "\n".join(classification_report_lines)
    return classification_report_content

def save_output(input_folder, classification_report_content):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    folder_name = os.path.splitext(os.path.basename(input_folder))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = os.path.join(output_dir, f"{timestamp}_{folder_name}")
    os.makedirs(timestamped_dir, exist_ok=True)

    classification_report_file = os.path.join(timestamped_dir, "classification-report.txt")
    with open(classification_report_file, "w") as f:
        f.write(classification_report_content)

    print(f"Image classification report saved to {classification_report_file}")

    if sys.platform == "win32":
        os.startfile(classification_report_file)
    else:
        open_command = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([open_command, classification_report_file])

def process_input(input_path):
    imagenet_labels_json = "imagenet-simple-labels.json"
    if not os.path.isfile(imagenet_labels_json):
        print(f"Error: The file '{imagenet_labels_json}' is missing. Please make sure it is in the same folder as the program.")
        sys.exit(1)

    with open(imagenet_labels_json, 'r') as f:
        imagenet_labels = json.load(f)

    if os.path.isdir(input_path):
        print("Starting image folder processing...")
        image_count, classification_counts, image_classifications, classification_images, processing_time = process_images(input_path, imagenet_labels)
        classification_report_content = generate_report(input_path, image_count, processing_time, classification_counts, image_classifications, classification_images)
        save_output(input_path, classification_report_content)
        print("Image folder processing completed.")
    else:
        print("Error: Invalid input. Please provide a folder containing images.")
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python vehicle_classification.py <input_folder>")
        sys.exit(1)

    input_path = sys.argv[1]
    process_input(input_path)

if __name__ == "__main__":
    main()