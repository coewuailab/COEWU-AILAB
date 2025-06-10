import machine
import network
import socket
import utime
import ntptime
from machine import UART, Pin, I2C
import json
from ufirebase import UFirebase
import gravity_pm25

# Configuration
WIFI_SSID = "UFPs-project" # UFPs-project
WIFI_PASSWORD = "coeai123"
ROOM = 'Cafe'

# Firebase project credentials
FIREBASE_URL = "https://ultrafine-particles-default-rtdb.asia-southeast1.firebasedatabase.app/"
FIREBASE_API_KEY = "AIzaSyABhPGdjCC4XEDK2x-Rfx-QsSX9aY59s-o"
FIREBASE_EMAIL = "poo.t2546@gmail.com"
FIREBASE_PASSWORD = "puwadech03"

# Pin Configuration
LED_PIN = 2
UART_TX = 18
UART_RX = 4

# I2C Configuration (สำหรับกรณีที่เซ็นเซอร์ใช้ I2C)
I2C_SCL = 22
I2C_SDA = 21

# ตั้งค่า LED
led = Pin(LED_PIN, Pin.OUT)

# เลือกวิธีการเชื่อมต่อกับเซ็นเซอร์ Gravity PM2.5 (UART หรือ I2C)
# ตรวจสอบข้อมูลจำเพาะของเซ็นเซอร์ที่คุณมีว่าใช้อินเตอร์เฟซแบบใด

# # สำหรับการเชื่อมต่อแบบ UART
# uart = UART(1, baudrate=9600, tx=UART_TX, rx=UART_RX)
# dust_sensor = gravity_pm25.GravityPM25(uart=uart)

# หรือสำหรับการเชื่อมต่อแบบ I2C (ให้ uncomment และแก้ไขถ้าต้องการใช้ I2C แทน)
i2c = I2C(0, scl=Pin(I2C_SCL), sda=Pin(I2C_SDA), freq=96000)
dust_sensor = gravity_pm25.GravityPM25(i2c=i2c)

# Initialize Firebase
firebase = UFirebase(FIREBASE_URL, FIREBASE_API_KEY, FIREBASE_EMAIL, FIREBASE_PASSWORD)

def connect_wifi():
    """Connect to WiFi with improved error handling and timeout"""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if not wlan.isconnected():
        print('connecting to network...', WIFI_SSID)
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        
        timeout = 0
        while not wlan.isconnected() and timeout < 60:
            print('.', end='')
            utime.sleep(1)
            timeout += 1
            
        if not wlan.isconnected():
            print('\nWiFi connection failed')
            return False
            
    print('\nConnected to WiFi: ', wlan.ifconfig()[0])
    return True

def sync_time():
    """Synchronize time with NTP server"""
    try:
        ntptime.settime()
        # Adjust to GMT+7
        rtc = machine.RTC()
        tm = utime.localtime()
        rtc.datetime((tm[0], tm[1], tm[2], tm[6], tm[3] + 7, tm[4], tm[5], 0))
        print('Time synced:', get_timestamp())
        return True
    except Exception as e:
        print('Time sync failed: ', e)
        return False

def get_time():
    """Get formatted time"""
    t = utime.localtime()
    return '{:02d}:{:02d}:{:02d}'.format(t[3], t[4], t[5])

def get_date():
    """Get formatted date"""
    t = utime.localtime()
    return '{:04d}-{:02d}-{:02d}'.format(t[0], t[1], t[2])

def get_timestamp():
    """Get formatted timestamp"""
    t = utime.localtime()
    return '{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'.format(
        t[0], t[1], t[2], t[3], t[4], t[5]
    )

def read_pm_sensors():
    """Read PM sensor with error handling"""
    try:
        # Read Gravity PM2.5 sensor
        success = dust_sensor.read()
        if success:
            pm1_0 = dust_sensor.pm1_0  # บางรุ่นอาจมีค่า PM1.0 ด้วย
            pm2_5 = dust_sensor.pm2_5
            pm10 = dust_sensor.pm10
        else:
            pm1_0 = pm2_5 = pm10 = 0.0
            
        return pm1_0, pm2_5, pm10
    
    except Exception as e:
        print('PM sensor read error:', e)
        return 0.0, 0.0, 0.0

def main_loop():
    """Main program loop with error handling"""
    print('Warming up sensors...')
    dust_sensor.wake()  # ปลุกเซ็นเซอร์ให้พร้อมทำงาน
    utime.sleep(30)
    
    while True:
        try:
            timestamp = get_timestamp()
            date = get_date()
            time = get_time()
            led.value(1)  # LED on
            
            # Read sensors - เฉพาะ PM2.5 เท่านั้น
            pm1_0, pm2_5, pm10 = read_pm_sensors()
            
            # Prepare data for Firebase
            data = {
                "Timestamp": timestamp,
                "PM1_0": pm1_0,
                "PM2_5": pm2_5,
                "PM10": pm10
            }
            
            print(timestamp, "PM1.0=", pm1_0, ", PM2.5=", pm2_5, ", PM10=", pm10)
            
            # Send to Firebase with retry mechanism
            retry_count = 0
            success = False
            while retry_count < 3 and not success:
                path = 'DFROBOT/' + ROOM + '/' + date + '/' + time
                success = firebase.put(path, data)
                if success:
                    print("Data sent successfully")
                else:
                    print("Failed to send data, attempt ", retry_count + 1, "/3")
                    retry_count += 1
                    if retry_count < 3:  # Don't wait after the last attempt
                        utime.sleep(2)  # Wait before retry 
                
            led.value(0)  # LED off
            utime.sleep(4)
            
        except Exception as e:
            print('Main loop error:', e)
            utime.sleep(10)

def start():
    """Initialize and start the program"""
    print('\nStarting PM2.5 monitoring for ', ROOM)
    
    # Initial setup
    if not connect_wifi():
        machine.reset()
    
    if not sync_time():
        machine.reset()
        
    # Initialize Firebase connection
    if not firebase.auth():
        print("Firebase authentication failed")
        machine.reset()
        
    try:
        main_loop()
    except KeyboardInterrupt:
        print('Program stopped by user')
    except Exception as e:
        print('Fatal error:', e)
        utime.sleep(10)
        machine.reset()

if __name__ == '__main__':
    start()
