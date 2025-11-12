import cv2
import numpy as np
import sky_mask
import math
import os
from openpyxl import Workbook
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

# 입출력 경로
VIDEO_DIR = "../inputs"
OUTPUT_DIR = "../outputs"

# 영상 포맷
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')

# 미리보기
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720

# 칼만 필터
FPS = 30.0
DELTA_T = 1.0 / FPS

# 칼만 필터 초기화 (roll)
state_dim_roll = 2
measurement_dim_roll = 1
kalman_filter_roll = cv2.KalmanFilter(state_dim_roll, measurement_dim_roll)
kalman_filter_roll.transitionMatrix = np.array([[1, DELTA_T], [0, 1]], np.float32)
kalman_filter_roll.measurementMatrix = np.array([[1, 0]], np.float32)
kalman_filter_roll.processNoiseCov = np.eye(state_dim_roll, dtype=np.float32) * 1e-3
kalman_filter_roll.measurementNoiseCov = np.eye(measurement_dim_roll, dtype=np.float32) * 1e-1
kalman_filter_roll.statePost = np.zeros((state_dim_roll, 1), np.float32)
kalman_filter_roll.errorCovPost = np.eye(state_dim_roll, dtype=np.float32) * 1

def fit_cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def sample_contour(contour, distance):
    sampled_points = [contour[0].squeeze()]
    cumulative_distance = 0
    for i in range(len(contour) - 1):
        p1 = contour[i].squeeze()
        p2 = contour[i+1].squeeze()
        segment_distance = np.linalg.norm(p2 - p1)
        cumulative_distance += segment_distance
        if cumulative_distance >= distance:
            sampled_points.append(p2)
            cumulative_distance -= distance
    return np.array(sampled_points)

def calculate_roll(coeffs):
    if coeffs is not None and len(coeffs) == 2:
        slope = coeffs[0]
        roll = int(math.degrees(math.atan(slope)) * 10)  # 10배 스케일링
        return roll
    return 0

def calculate_pitch_for_simulator(sky_ratio):
    ground_ratio = 1.0 - sky_ratio
    pitch = (ground_ratio - 0.5) / 0.5 * 100  # -100 ~ 100 스케일
    return int(pitch)

def create_output_directories(output_dir, folder_name):
    os.makedirs(os.path.join(output_dir, folder_name, "videos"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, folder_name, "xlsx"), exist_ok=True)

def interpolate_nan_values(data):
    data = np.array(data, dtype=np.float64)
    nans = np.isnan(data)
    if np.all(nans):
        return np.zeros_like(data)
    x = np.arange(len(data))
    f = interp1d(x[~nans], data[~nans], kind='linear', fill_value="extrapolate")
    return f(x)

def clamp_values(data, min_val=-100, max_val=100):
    return np.clip(data, min_val, max_val)

# 비디오 폴더 리스트
input_folders = [""]

for folder_name in input_folders:
    folder_path = os.path.join(VIDEO_DIR, folder_name)
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(VIDEO_EXTENSIONS)]

    create_output_directories(OUTPUT_DIR, folder_name)

    for video_file in video_files:
        VIDEO_PATH = os.path.join(folder_path, video_file)
        base_name = os.path.splitext(video_file)[0]
        OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, folder_name, "videos", f"{base_name}.mp4")
        XLSX_OUTPUT_PATH_ORIGINAL = os.path.join(OUTPUT_DIR, folder_name, "xlsx", f"{base_name}_original.xlsx")
        XLSX_OUTPUT_PATH_REFINED = os.path.join(OUTPUT_DIR, folder_name, "xlsx", f"{base_name}.xlsx")

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"❌ {VIDEO_PATH} 비디오를 열 수 없습니다.")
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

        workbook_orig = Workbook()
        sheet_orig = workbook_orig.active
        sheet_orig.append(['Frame No', 'Roll Data', 'Pitch Data', 'Wind Data'])

        workbook_refined = Workbook()
        sheet_refined = workbook_refined.active
        sheet_refined.append(['Frame No', 'Roll Data', 'Pitch Data', 'Wind Data'])

        frame_number = 0
        sky_ratios = []
        roll_values = []
        pitch_values = []

        kalman_filter_roll.statePost = np.zeros((state_dim_roll, 1), np.float32)
        kalman_filter_roll.errorCovPost = np.eye(state_dim_roll, dtype=np.float32) * 1

        user_choice = input("Select occluder set (1: 드론/비행기, 2: 레이싱, 3: 보트): ").strip()
        scale = float(input(f"Enter pitch scale (default = 200): ") or 200)

        sky_mask.set_occlude_choice(user_choice)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1

            sky_mask_output_resized = sky_mask.get_sky_mask(frame)
            sky_mask_fullres = cv2.resize(sky_mask_output_resized, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

            edges = cv2.Canny(sky_mask_fullres, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            blended = frame.copy()

            rows, cols = sky_mask_fullres.shape[:2]
            sky_pixels = np.sum(sky_mask_fullres == 255)
            total_pixels = rows * cols
            sky_ratio = sky_pixels / total_pixels if total_pixels > 0 else 0
            sky_ratios.append(sky_ratio)

            sampling_distance = int(max(frame_width, frame_height) / 100)

            raw_roll = 0
            kalman_roll_value = None

            for contour in contours:
                if len(contour) >= 10:
                    sampled_points = sample_contour(contour, sampling_distance)
                    if len(sampled_points) >= 4:
                        x_coords = sampled_points[:, 0]
                        y_coords = sampled_points[:, 1]
                        try:
                            popt, _ = curve_fit(fit_cubic, x_coords, y_coords, p0=[0,0,0,frame_height//2])
                            a, b, c, d = popt
                            x_curve = np.linspace(np.min(x_coords), np.max(x_coords), 100).astype(int)
                            y_curve = (a*x_curve**3 + b*x_curve**2 + c*x_curve + d).astype(int)

                            # 곡선
                            for i in range(len(x_curve) - 1):
                                pt1 = (x_curve[i], y_curve[i])
                                pt2 = (x_curve[i + 1], y_curve[i + 1])
                                if 0 <= pt1[0] < cols and 0 <= pt1[1] < rows and 0 <= pt2[0] < cols and 0 <= pt2[
                                    1] < rows:
                                    cv2.line(blended, pt1, pt2, (0, 255, 0), 2)

                            # 직선
                            coeffs_line = np.polyfit(x_curve, y_curve, 1)
                            slope = coeffs_line[0]
                            intercept = coeffs_line[1]
                            x1, x2 = 0, cols - 1
                            y1 = int(slope * x1 + intercept)
                            y2 = int(slope * x2 + intercept)
                            if 0 <= y1 < rows and 0 <= y2 < rows:
                                cv2.line(blended, (x1, y1), (x2, y2), (255, 0, 0), 3)

                            raw_roll = calculate_roll(coeffs_line)

                            # 칼만 필터 업데이트
                            kalman_filter_roll.predict()
                            kalman_filter_roll.correct(np.array([[np.float32(raw_roll)]]))
                            kalman_roll_value = kalman_filter_roll.statePost[0,0]

                        except Exception as e:
                            kalman_roll_value = None
                            raw_roll = 0
                    else:
                        kalman_roll_value = None
                        raw_roll = 0
                else:
                    kalman_roll_value = None
                    raw_roll = 0

            simulator_pitch = calculate_pitch_for_simulator(sky_ratio)

            roll_values.append(kalman_roll_value if kalman_roll_value is not None else np.nan)
            pitch_values.append(simulator_pitch)

            if isinstance(kalman_roll_value, (int, float, np.floating, np.integer)):
                cv2.putText(blended, f"Roll Kalman: {int(kalman_roll_value)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

            if isinstance(simulator_pitch, (int, float, np.floating, np.integer)):
                cv2.putText(blended, f"Pitch Sim: {int(simulator_pitch)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 255), 2)

            overlay = np.zeros_like(frame)
            overlay[sky_mask_fullres == 255] = (255, 200, 100)
            blended = cv2.addWeighted(blended, 1, overlay, 0.3, 0)

            # 미리보기
            resized_blended = cv2.resize(blended, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
            # cv2.imshow("Result", resized_blended)
            out.write(blended)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # 이상치 판단 및 보간 처리
        sky_ratios_array = np.array(sky_ratios)

        # IQR 기반 이상치 탐지
        q1 = np.percentile(sky_ratios_array, 25)
        q3 = np.percentile(sky_ratios_array, 75)
        iqr = q3 - q1

        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr

        iqr_mask = (sky_ratios_array < iqr_lower) | (sky_ratios_array > iqr_upper)

        # 프레임 간 급변 탐지
        smoothed = uniform_filter1d(sky_ratios_array, size=5)
        diff = np.abs(sky_ratios_array - smoothed)
        jump_threshold = 0.10  # 변화율 임계값 (10%)
        jump_mask = diff > jump_threshold

        # 종합 이상치 마스크
        outlier_mask = iqr_mask | jump_mask
        pitch_outlier_mask = jump_mask

        # 이상치 제거 후 baseline 계산
        valid_sky_ratios = sky_ratios_array[~outlier_mask]
        baseline_sky_ratio = np.mean(valid_sky_ratios)

        # pitch 계산
        pitch_array = (baseline_sky_ratio - sky_ratios_array) * scale

        # roll/pitch 값에 이상치 적용
        roll_array = np.array(roll_values, dtype=np.float64)
        pitch_array = np.array(pitch_array, dtype=np.float64)

        # 마스킹
        roll_array[outlier_mask] = np.nan
        pitch_array[pitch_outlier_mask] = np.nan

        # 보간
        roll_interpolated = interpolate_nan_values(roll_array)
        pitch_interpolated = interpolate_nan_values(pitch_array)

        # 클램핑
        roll_interpolated = clamp_values(roll_interpolated)
        pitch_interpolated = clamp_values(pitch_interpolated)

        # 원본 데이터 저장
        for i in range(len(roll_array)):
            roll_write = int(roll_array[i]) if not np.isnan(roll_array[i]) else ""
            pitch_write = int(pitch_array[i]) if not np.isnan(pitch_array[i]) else ""
            wind_val = 0

            sheet_orig.append([i + 1, roll_write, pitch_write, wind_val])

        workbook_orig.save(XLSX_OUTPUT_PATH_ORIGINAL)

        # 보간 데이터 저장
        for i in range(len(roll_interpolated)):
            roll_write = int(np.clip(roll_interpolated[i], -100, 100))
            pitch_write = int(pitch_interpolated[i])
            wind_val = 0

            sheet_refined.append([i + 1, roll_write, pitch_write, wind_val])

        workbook_refined.save(XLSX_OUTPUT_PATH_REFINED)

        print(f"✅ 완료: {video_file}, 결과 영상 및 엑셀 저장 완료")
