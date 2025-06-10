import machine
import network
import socket
import utime
import ntptime
from machine import UART, Pin
import dht
import sds011
import json
from ufirebase import UFirebase

# Configuration
WIFI_SSID = "ENGIOT"
WIFI_PASSWORD = "coeai123"
ROOM = 'Lab'

# Firebase project credentials
FIREBASE_URL = "https://ultrafine-particles-default-rtdb.asia-southeast1.firebasedatabase.app/"
FIREBASE_API_KEY = "AIzaSyABhPGdjCC4XEDK2x-Rfx-QsSX9aY59s-o"
FIREBASE_EMAIL = "poo.t2546@gmail.com"
FIREBASE_PASSWORD = "puwadech03"

# Pin Configuration
DHT_PIN = 27
LED_PIN = 2
UART_TX = 18
UART_RX = 4

# Initialize sensors
dht_sensor = dht.DHT22(machine.Pin(DHT_PIN))
uart = UART(1, baudrate=9600, tx=UART_TX, rx=UART_RX)
dust_sensor = sds011.SDS011(uart)
led = Pin(LED_PIN, Pin.OUT)

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

def mean(values):
    """Calculate mean of values with error handling"""
    if not values:
        return 0.0
    return sum(values) / len(values)

def read_dht_sensors():
    """Read all sensors with error handling"""
    try:
        # Read DHT22
        dht_sensor.measure()
        temp = dht_sensor.temperature()
        hum = dht_sensor.humidity()
        
    except Exception as e:
        print('DHT read error:', e)
        temp = hum = 0.0
    
    return temp, hum

def read_sds_sensors():
    # Read SDS011
    try:
        dust_sensor.read()
        if dust_sensor.packet_status:
            pm25 = dust_sensor.pm25
            pm10 = dust_sensor.pm10
        else:
            pm10 = pm25 = 0.0
            
        return pm10, pm25
    
    except Exception as e:
        print('SDS011 read error:', e)
        pm25 = pm10 = 0.0

    return pm10, pm25

def main_loop():
    """Main program loop with error handling"""
    dust_sensor.wake()
    print('Warming up sensors...')
    utime.sleep(30)
    
    while True:
        try:
            timestamp = get_timestamp()
            date = get_date()
            time = get_time()
            led.value(1)  # LED on
            
            # Read sensors
            temp, hum = read_dht_sensors()
            pm10, pm25 = read_sds_sensors()
            
            # Prepare data for Firebase
            data = {
                "Timestamp": timestamp,
                "IndoorTemperature": temp,
                "IndoorHumidity": hum,
                "PM10": pm10,
                "PM25": pm25
            }
            
            print(timestamp, "T=", temp, "°C, H=", hum, "%, PM10=", pm10, ", PM2.5=", pm25)
            
            # Send to Firebase with retry mechanism
            retry_count = 0
            success = False
            while retry_count < 3 and not success:
                path = 'RAWdata/' + ROOM + '/' + date + '/' + time
                success = firebase.put(path, data)
                if success:
                    print("Data sent successfully")
                else:
                    print("Failed to send data, attempt ", retry_count + 1, "/3")
                    retry_count += 1
                    if retry_count < 3:
                        utime.sleep(2)
                
            led.value(0)  # LED off
            utime.sleep(4)
            
        except Exception as e:
            print('Main loop error:', e)
            utime.sleep(10)

def start():
    """Initialize and start the program"""
    print('\nStarting sensor monitoring for ', ROOM)
    
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