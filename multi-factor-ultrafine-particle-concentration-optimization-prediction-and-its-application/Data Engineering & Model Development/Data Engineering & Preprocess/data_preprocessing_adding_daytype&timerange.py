import pandas as pd
import datetime

# อ่านไฟล์ CSV
print("Reading CSV file...")
df = pd.read_csv('processed_data_selected_features_1min_cleaned.csv')

# ตรวจสอบรูปแบบของคอลัมน์ Time โดยดูข้อมูลตัวอย่าง
print("Sample data in Time column:")
print(df['Time'].head())

# แปลงคอลัมน์ Time เป็น datetime
print("Converting date format...")
try:
    # ลองใช้ pandas auto-detection
    df['Time'] = pd.to_datetime(df['Time'])
    print("Automatic date conversion successful")
except:
    # หากไม่สำเร็จ ให้ลองใช้รูปแบบอื่น
    print("Automatic conversion failed, trying other formats...")
    try:
        df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')
    except:
        try:
            df['Time'] = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M:%S')
        except:
            try:
                df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M:%S')
            except:
                print("Cannot convert dates. Please check date format in CSV file")
                exit(1)

# สร้างคอลัมน์วันที่ (เฉพาะวันที่ ไม่มีเวลา) และคอลัมน์วันในสัปดาห์
df['date'] = df['Time'].dt.date
df['day_of_week'] = df['Time'].dt.dayofweek  # 0 = Monday, 6 = Sunday
df['day_name'] = df['Time'].dt.day_name()  # Day name in English

# เพิ่มคอลัมน์ daytype
# วันจันทร์(0) ถึงวันศุกร์(4) เป็น workday, วันเสาร์(5) และวันอาทิตย์(6) เป็น weekend
df['daytype'] = df['day_of_week'].apply(lambda x: 'workday' if x < 5 else 'weekend')

# ดึงชั่วโมงจากเวลา
df['hour'] = df['Time'].dt.hour


# เพิ่มคอลัมน์ time_range
def get_time_range(hour):
    if 6 <= hour < 9:
        return 'early_morning'
    elif 9 <= hour < 11:
        return 'morning'
    elif 11 <= hour < 13:
        return 'early_afternoon'
    elif 13 <= hour < 15:
        return 'afternoon'
    elif 15 <= hour < 18:
        return 'late_afternoon'
    else:
        return 'other_time'  # สำหรับชั่วโมงที่ไม่อยู่ในช่วงที่กำหนด

# delete other time ranges if not needed
df = df[df['hour'].between(6, 17)]  # เก็บเฉพาะชั่วโมง 6 ถึง 17

df['time_range'] = df['hour'].apply(get_time_range)

# =========== สร้าง Log สำหรับตรวจสอบความถูกต้อง ===========

# สร้าง log ในรูปแบบ txt โดยระบุ encoding เป็น utf-8
with open("daytype_validation_log.txt", "w", encoding="utf-8") as f:
    f.write("===== Day Type Validation Report =====\n")
    f.write("=====================================\n\n")

    # สรุปจำนวนวันทั้งหมด
    f.write("1. Summary\n")
    f.write("----------\n")
    total_days = df['date'].nunique()
    f.write(f"Total days in dataset: {total_days} days\n")
    workday_count = df[df['daytype'] == 'workday']['date'].nunique()
    weekend_count = df[df['daytype'] == 'weekend']['date'].nunique()
    f.write(f"Workdays: {workday_count} days\n")
    f.write(f"Weekends: {weekend_count} days\n\n")

    # สรุปจำนวนวันแยกตามวันในสัปดาห์
    f.write("2. Days by weekday\n")
    f.write("-----------------\n")
    day_counts = df[['date', 'day_name']].drop_duplicates().groupby('day_name').size()
    for day, count in day_counts.items():
        f.write(f"{day}: {count} days\n")
    f.write("\n")

    # รายการวันที่เป็น workday
    f.write("3. Workday dates\n")
    f.write("---------------\n")
    workdays = df[df['daytype'] == 'workday'][['date', 'day_name']].drop_duplicates().sort_values('date')
    for _, row in workdays.iterrows():
        f.write(f"{row['date']} - {row['day_name']}\n")
    f.write("\n")

    # รายการวันที่เป็น weekend
    f.write("4. Weekend dates\n")
    f.write("---------------\n")
    weekends = df[df['daytype'] == 'weekend'][['date', 'day_name']].drop_duplicates().sort_values('date')
    for _, row in weekends.iterrows():
        f.write(f"{row['date']} - {row['day_name']}\n")
    f.write("\n")

    # สรุปข้อมูล time_range
    f.write("5. Time range summary\n")
    f.write("-------------------\n")
    time_range_counts = df['time_range'].value_counts()
    for time_range, count in time_range_counts.items():
        f.write(f"{time_range}: {count} records\n")

    # ตัวอย่างข้อมูลแต่ละช่วงเวลา
    f.write("\n6. Sample data for each time range\n")
    f.write("--------------------------------\n")
    for time_range in df['time_range'].unique():
        f.write(f"\nSamples for {time_range}:\n")
        sample = df[df['time_range'] == time_range].head(3)
        for _, row in sample.iterrows():
            f.write(f"  {row['Time']} - {row['day_name']} - {row['daytype']}\n")

print("\nLog file 'daytype_validation_log.txt' created successfully")

# ลบคอลัมน์ที่ใช้เป็นช่วงคราว
df = df.drop(['date', 'day_of_week', 'day_name', 'hour'], axis=1)

# แสดงข้อมูลตัวอย่างหลังการเพิ่มคอลัมน์
print("\nSample data after adding columns:")
print(df[['Time', 'daytype', 'time_range']].head(10))

df = df.dropna()
# บันทึกเป็นไฟล์ CSV ใหม่
print("\nSaving file...")
df.to_csv('processed_data_selected_features_1min_cleaned(Completed).csv', index=False)
print("Done! File saved as 'processed_data_selected_features_1min_cleaned(Completed).csv'")