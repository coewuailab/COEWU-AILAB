"""
สคริปต์สำหรับการเรียกใช้งาน RetrainTopModels
ใช้สำหรับเรียกใช้โค้ดการเทรนซ้ำโมเดลที่ดีที่สุด 10 อันดับแล้วเทรนซ้ำแต่ละโมเดล 10 รอบ
"""

import os
import sys
from datetime import datetime
from RNN_Retrain_Top_Models import RetrainTopModels

# ตรวจสอบว่ามีการระบุพาธของโฟลเดอร์ผลลัพธ์หรือไม่
if len(sys.argv) > 1:
    RUN_FOLDER = sys.argv[1]
else:
    # ถ้าไม่ได้ระบุให้ใช้โฟลเดอร์ล่าสุดในโฟลเดอร์ Bayesian Tuning
    BASE_DIR = "PM01_RNN_Bayesian_Tuning"

    # ค้นหาโฟลเดอร์ที่มีชื่อขึ้นต้นด้วย "run_" แล้วเรียงตามวันที่
    run_folders = [folder for folder in os.listdir(BASE_DIR) if folder.startswith("run_")]
    if run_folders:
        run_folders.sort(reverse=True)  # เรียงจากใหม่ไปเก่า
        RUN_FOLDER = os.path.join(BASE_DIR, run_folders[0])
    else:
        print("ไม่พบโฟลเดอร์ run_ ใน", BASE_DIR)
        sys.exit(1)

# ตรวจสอบว่ามีโฟลเดอร์ top_models หรือไม่
TOP_MODELS_DIR = os.path.join(RUN_FOLDER, "top_models")
if not os.path.exists(TOP_MODELS_DIR):
    print(f"ไม่พบโฟลเดอร์ {TOP_MODELS_DIR}")
    sys.exit(1)

# ตรวจสอบไฟล์ all_models_summary.json
TOP_MODELS_SUMMARY_PATH = os.path.join(TOP_MODELS_DIR, "all_models_summary.json")
if not os.path.exists(TOP_MODELS_SUMMARY_PATH):
    print(f"ไม่พบไฟล์ {TOP_MODELS_SUMMARY_PATH}")
    sys.exit(1)

# พาธของไฟล์ CSV ที่ใช้ในการเทรน
DATA_PATH = "Merge_81Days_1min_with_daytypes.csv"
if not os.path.exists(DATA_PATH):
    print(f"ไม่พบไฟล์ {DATA_PATH}")
    sys.exit(1)

# กำหนดจำนวนรอบที่ต้องการเทรนซ้ำและจำนวนโมเดล
NUM_RETRAIN = 10
NUM_TOP_MODELS = 10

# สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"PM01_RNN_Retrain_Top{NUM_TOP_MODELS}_{timestamp}"

print(f"{'='*80}")
print(f"เริ่มการเทรนซ้ำโมเดลที่ดีที่สุด {NUM_TOP_MODELS} อันดับแรก โดยเทรนซ้ำ {NUM_RETRAIN} รอบต่อโมเดล")
print(f"ใช้ข้อมูลจาก: {TOP_MODELS_SUMMARY_PATH}")
print(f"ไฟล์ข้อมูล: {DATA_PATH}")
print(f"โฟลเดอร์ผลลัพธ์: {OUTPUT_DIR}")
print(f"{'='*80}")

# เรียกใช้งานฟังก์ชัน RetrainTopModels
try:
    # สร้างอินสแตนซ์และเรียกใช้งานโดยตรง
    retrain = RetrainTopModels(
        top_models_summary_path=TOP_MODELS_SUMMARY_PATH,
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        num_retrain=NUM_RETRAIN,
        num_top_models=NUM_TOP_MODELS
    )

    results = retrain.retrain_all_models()

    print(f"\n{'=' * 80}")
    print("เสร็จสิ้นการเทรนโมเดลซ้ำทั้งหมด")
    print(f"ผลลัพธ์ถูกบันทึกที่: {OUTPUT_DIR}")
    print(f"{'=' * 80}")

except Exception as e:
    print(f"\nเกิดข้อผิดพลาด: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)