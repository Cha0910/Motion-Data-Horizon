import cv2
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "nvidia/segformer-b1-finetuned-ade-512-512"
PROCESSOR = SegformerImageProcessor.from_pretrained(MODEL_NAME)
MODEL = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(DEVICE)
if DEVICE.type == 'cuda':
    MODEL = MODEL.half()

OCCLUDER_NAME_GROUPS = {
    "1": ["building", "house", "skyscraper", "tower", "bridge", "airplane"],  # 드론, 비행기
    "2": ["airplane", "building", "skyscraper", "person", "tree", "streetlight", "pole", "fence", "bridge", "mountain"],  # 레이싱(도로/트랙)
    "3": ["building", "skyscraper", "boat", "ship", "bridge", "tree", "house", "person", "palm", "mountain"],  # 보트(수상)
}

CLASS_NAMES = MODEL.config.id2label
SKY_CLASS_INDEX = next((id for id, name in CLASS_NAMES.items() if name == 'sky'), 2)

# ============================== 유틸 함수 ==============================

def resize_to_input(image, size=(512, 512)):
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def apply_clahe_rgb(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def fill_below_sky_horizon(mask):
    horizon_line = np.zeros(mask.shape[1], dtype=np.int32)
    for x in range(mask.shape[1]):
        ys = np.where(mask[:, x] == 255)[0]
        horizon_line[x] = ys.max() if len(ys) > 0 else 0
    filled_mask = mask.copy()
    for x in range(mask.shape[1]):
        y_max = horizon_line[x]
        if y_max > 0:
            filled_mask[:y_max, x] = 255
    return filled_mask

def refine_sky_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels > 1:
        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned = np.where(labels == max_label, 255, 0).astype(np.uint8)

    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=5)
    cleaned = cv2.medianBlur(cleaned, 5)
    return np.where(cleaned > 127, 255, 0).astype(np.uint8)

def extend_mask_based_on_edges(mask, extend_pixels=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, extend_pixels))
    return cv2.dilate(mask, kernel, iterations=1)

def fill_occluded_sky(sky_mask, pred_class_map, occluder_ids, dilation_iter=5):
    pred_resized = cv2.resize(pred_class_map.astype(np.uint8), (sky_mask.shape[1], sky_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    occluder_mask = np.isin(pred_resized, occluder_ids).astype(np.uint8) * 255

    # 팽창
    sky_dilated = cv2.dilate(sky_mask, np.ones((3, 3), np.uint8), iterations=dilation_iter)

    height, width = sky_mask.shape

    bottom_ys = []
    for x in range(width):
        ys = np.where(sky_mask[:, x] == 255)[0]
        if len(ys) > 0:
            bottom_ys.append(ys.max())
    if len(bottom_ys) == 0:
        avg_bottom = 0
    else:
        avg_bottom = int(np.mean(bottom_ys))

    # avg_bottom 아래는 팽창 안 하도록 마스크 제한
    sky_dilated[avg_bottom + 1:, :] = 0

    # occluder와 맞닿은 영역 찾기
    touching_occluders = cv2.bitwise_and(sky_dilated, occluder_mask)

    contours, _ = cv2.findContours(touching_occluders, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    occluder_fill_mask = np.zeros_like(sky_mask)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(occluder_fill_mask, (x, y), (x + w, y + h), 255, -1)

    filled = cv2.bitwise_or(sky_mask, occluder_fill_mask)
    return filled

def get_occluder_ids(class_names, choice):
    if choice not in OCCLUDER_NAME_GROUPS:
        raise ValueError(f"Invalid choice: {choice}")
    selected_names = OCCLUDER_NAME_GROUPS[choice]
    occluder_ids = [cid for cid, cname in class_names.items() if cname in selected_names]
    return occluder_ids

# ============================== 클래스 ==============================

class SkyMaskExtractor:
    def __init__(self):
        self.prev_frame = None
        self.prev_mask = None
        self.occluder_class_ids = None  # 초기에는 None

    def set_choice(self, choice):
        """ occluder set 설정 (1, 2, 3 중 하나) """
        self.occluder_class_ids = get_occluder_ids(CLASS_NAMES, choice)

    def get_mask(self, current_frame):
        if self.occluder_class_ids is None:
            raise ValueError("Occluder set not selected. Call set_choice(choice) first.")

        enhanced = apply_clahe_rgb(current_frame)
        resized = resize_to_input(enhanced)

        pil_image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        inputs = PROCESSOR(images=pil_image, return_tensors="pt").to(DEVICE)
        if DEVICE.type == 'cuda':
            inputs = {k: v.half() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = MODEL(**inputs)

        pred_class_map = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()
        sky_mask = (pred_class_map == SKY_CLASS_INDEX).astype(np.uint8) * 255
        raw_mask_resized = cv2.resize(sky_mask, (current_frame.shape[1], current_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 하늘 아래는 잘라내기
        clipped_mask = fill_below_sky_horizon(raw_mask_resized)

        # 가려진 하늘 복원
        filled_mask = fill_occluded_sky(clipped_mask, pred_class_map, self.occluder_class_ids)

        # 후처리
        refined_mask = refine_sky_mask(filled_mask)

        # Temporal Smoothing
        smoothed_mask = self.temporal_smooth_optical_flow(refined_mask, current_frame)

        # 마지막 정리 + 경계 확장
        final_mask = refine_sky_mask(smoothed_mask)
        final_mask = extend_mask_based_on_edges(final_mask, extend_pixels=10)

        self.prev_frame = current_frame.copy()
        self.prev_mask = refined_mask.copy()

        return final_mask

    def temporal_smooth_optical_flow(self, current_mask, current_frame, alpha=0.8):
        if self.prev_frame is None or self.prev_mask is None:
            return current_mask

        if self.prev_frame.shape[:2] != current_frame.shape[:2]:
            self.prev_frame = None
            self.prev_mask = None
            return current_mask

        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        h, w = current_mask.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        warped_coords_x = (grid_x + flow[..., 0]).astype(np.float32)
        warped_coords_y = (grid_y + flow[..., 1]).astype(np.float32)

        warped_prev_mask = cv2.remap(self.prev_mask, warped_coords_x, warped_coords_y,
                                     interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        blended = cv2.addWeighted(current_mask.astype(np.float32), alpha,
                                  warped_prev_mask.astype(np.float32), 1 - alpha, 0)
        return np.clip(blended, 0, 255).astype(np.uint8)

# ============================== 외부 인터페이스 ==============================

_EXTRACTOR = SkyMaskExtractor()

def get_sky_mask(frame):
    return _EXTRACTOR.get_mask(frame)

def set_occlude_choice(choice):
    _EXTRACTOR.set_choice(choice)