import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import glob
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_SIZE = 512
DATA_DIR = "../uploads/students_augmented"  # Directory containing original class folders

# IResNet Backbone Definition
__all__ = ['iresnet100']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

    def forward(self, x):
        return self.forward_impl(x)

class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x

def iresnet100(**kwargs):
    return IResNet(IBasicBlock, [3, 13, 30, 3], **kwargs)

# ArcFace Loss
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weight)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = F.one_hot(labels, num_classes=self.out_features).float().to(DEVICE)
        target_cos = torch.cos(theta + self.m)
        output = self.s * (one_hot * target_cos + (1.0 - one_hot) * cosine)
        return F.cross_entropy(output, labels)

# Face Classifier Model
class FaceClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# Image Loader
class ImageLoader:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def load_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image).unsqueeze(0)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

def get_class_names():
    """Get class names from DATA_DIR."""
    class_names = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    class_names.sort()
    return class_names

def load_model(class_names):
    """Load the trained face recognition model."""
    num_classes = len(class_names)
    backbone = iresnet100(num_features=EMBEDDING_SIZE)
    model = FaceClassifier(backbone).to(DEVICE)
    criterion = ArcFaceLoss(EMBEDDING_SIZE, num_classes, s=30.0, m=0.50).to(DEVICE)

    try:
        checkpoint = torch.load("models/fasmodel.pt", map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        criterion.load_state_dict(checkpoint['criterion_state_dict'])
        print(f"Model loaded successfully with {num_classes} classes")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    model.eval()
    return model, criterion

def check_existing_logs(output_folder, stream_id):
    """Check existing logs to determine recognized and unrecognized tracks."""
    log_file = os.path.join(output_folder, f"stream_{stream_id}", "attendance_log.txt")
    recognized_images = set()
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 4:  # Ensure all fields are present
                    track_id = parts[2]  # TrackID is now the third column
                    filename = parts[1].split('/')[-1]  # Extract filename from Roll No if it includes path
                    recognized_images.add(f"{track_id}/{filename}")

    faces_folder = os.path.join(output_folder, f"stream_{stream_id}", "faces")
    total_images = 0
    track_folders = [d for d in os.listdir(faces_folder) if d.startswith("track_")] if os.path.exists(faces_folder) else []
    for track in track_folders:
        track_path = os.path.join(faces_folder, track)
        image_files = glob.glob(os.path.join(track_path, "*.jpg")) + \
                      glob.glob(os.path.join(track_path, "*.jpeg")) + \
                      glob.glob(os.path.join(track_path, "*.png"))
        total_images += len(image_files)

    unrecognized_images = total_images - len(recognized_images)
    print(f"Log analysis for stream_{stream_id}: {len(recognized_images)} images already recognized, {unrecognized_images} images to process")
    return recognized_images

def predict_face(model, criterion, image_tensor, class_names, confidence_threshold=0.7):
    """Predict the identity of a face in an image and return confidence."""
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        embeddings = model(image_tensor)
        logits = F.linear(F.normalize(embeddings), F.normalize(criterion.weight))
        logits = logits * criterion.s
        probabilities = F.softmax(logits, dim=1)

        confidence, predicted_idx = torch.max(probabilities, 1)
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()

        if confidence >= confidence_threshold:
            return class_names[predicted_idx], confidence
        else:
            return "Unknown", confidence

def generate_attendance_summary(output_folder, stream_id):
    """Generate attendance summary based on voting (mode) from logs."""
    log_file = os.path.join(output_folder, f"stream_{stream_id}", "attendance_log.txt")
    summary_file = os.path.join(output_folder, f"stream_{stream_id}", "attendance_summary.txt")
    
    if not os.path.exists(log_file):
        print(f"No log file found at {log_file}")
        return

    # Read all predictions from the log file
    track_predictions = {}
    with open(log_file, 'r') as f:
        for line in f.readlines()[1:]:  # Skip header
            parts = line.strip().split(',')
            if len(parts) >= 4:
                timestamp, roll_no, track_id, stream_id = parts[:4]
                if track_id not in track_predictions:
                    track_predictions[track_id] = []
                track_predictions[track_id].append(roll_no)

    # Calculate mode for each track
    summary_data = []
    for track_id, predictions in sorted(track_predictions.items(), key=lambda x: int(x[0].split('_')[1])):
        if not predictions:
            continue
            
        # Calculate mode (most frequent prediction)
        unique, counts = np.unique(predictions, return_counts=True)
        mode_index = np.argmax(counts)
        mode_prediction = unique[mode_index]
        
        # Get first timestamp for this track
        first_timestamp = None
        with open(log_file, 'r') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 4 and parts[2] == track_id:
                    first_timestamp = parts[0]
                    break
        
        # Determine if the person came in (mode matches track ID number if track IDs contain roll numbers)
        came_in = "True" if mode_prediction != "Unknown" else "False"
        
        summary_data.append({
            'track_id': track_id,
            'roll_number': track_id.split('_')[1] if '_' in track_id else track_id,  # Assuming track ID contains roll number
            'mode_prediction': mode_prediction,
            'timestamp': first_timestamp,
            'came_in': came_in
        })

    # Write summary file
    with open(summary_file, 'w') as f:
        f.write("Track ID,Roll number,Mode Prediction,timestamp,came_in\n")
        for data in summary_data:
            line = f"{data['track_id']},{data['roll_number']},{data['mode_prediction']},{data['timestamp']},{data['came_in']}\n"
            f.write(line)
    
    print(f"\nAttendance Summary (Voting-based) for stream_{stream_id}:")
    print("Track ID,Roll number,Mode Prediction,timestamp,came_in")
    for data in summary_data:
        print(f"{data['track_id']},{data['roll_number']},{data['mode_prediction']},{data['timestamp']},{data['came_in']}")

def process_tracks(output_folder, stream_id, model, criterion, class_names, recognized_images):
    """Process all tracks and save detailed results with simplified log structure."""
    image_loader = ImageLoader()
    faces_folder = os.path.join(output_folder, f"stream_{stream_id}", "faces")
    log_file = os.path.join(output_folder, f"stream_{stream_id}", "attendance_log.txt")
    
    if not os.path.exists(faces_folder):
        print(f"Error: Faces folder {faces_folder} does not exist")
        return

    track_folders = [d for d in os.listdir(faces_folder) if d.startswith("track_")]
    track_folders.sort(key=lambda x: int(x.split("_")[1]))

    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("Timestamp,Roll No,TrackID,StreamID\n")

    for track in tqdm(track_folders, desc=f"Processing tracks in stream_{stream_id}"):
        track_path = os.path.join(faces_folder, track)
        image_files = glob.glob(os.path.join(track_path, "*.jpg")) + \
                      glob.glob(os.path.join(track_path, "*.jpeg")) + \
                      glob.glob(os.path.join(track_path, "*.png"))
        image_files.sort()

        if not image_files:
            print(f"No images found in {track_path}")
            continue

        for img_path in image_files:
            filename = os.path.basename(img_path)
            image_key = f"{track}/{filename}"

            if image_key in recognized_images:
                continue

            image_tensor = image_loader.load_image(img_path)
            if image_tensor is None:
                print(f"Failed to load image: {img_path}")
                continue

            person, confidence = predict_face(model, criterion, image_tensor, class_names)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(log_file, 'a') as f:
                f.write(f"{timestamp},{person},{track},{stream_id}\n")

    print(f"Results appended to {log_file}")
    generate_attendance_summary(output_folder, stream_id)

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Inference")
    parser.add_argument("--output_folder", required=True, type=str, help="Output folder from alignmentvideo.py containing stream_* directories")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        print(f"Error: Output folder {args.output_folder} does not exist")
        return

    class_names = get_class_names()
    print(f"Found {len(class_names)} classes: {', '.join(class_names)}")

    model, criterion = load_model(class_names)

    stream_folders = [d for d in os.listdir(args.output_folder) if d.startswith("stream_")]
    if not stream_folders:
        print(f"No stream folders found in {args.output_folder}")
        return

    for stream_folder in stream_folders:
        stream_id = int(stream_folder.split("_")[1])
        recognized_images = check_existing_logs(args.output_folder, stream_id)
        process_tracks(args.output_folder, stream_id, model, criterion, class_names, recognized_images)

if __name__ == "__main__":
    main()