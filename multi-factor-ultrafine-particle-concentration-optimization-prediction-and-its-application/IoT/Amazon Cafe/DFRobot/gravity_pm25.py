# gravity_pm25.py - Driver for DFRobot Gravity PM2.5 Air Quality Sensor

import utime
from machine import UART, Pin, I2C

# รองรับทั้งการเชื่อมต่อแบบ UART และ I2C
class GravityPM25:
    """
    DFRobot Gravity PM2.5 Air Quality Sensor Driver
    รองรับทั้งการเชื่อมต่อแบบ UART และ I2C
    """
    
    # Default I2C address
    DEFAULT_I2C_ADDR = 0x19
    
    # Commands and constants
    CMD_READ_PARTICLE = 0x05
    
    def __init__(self, uart=None, i2c=None, i2c_addr=DEFAULT_I2C_ADDR):
        """
        Initialize the sensor with either UART or I2C interface
        
        :param uart: ถ้าใช้ UART ให้ส่ง UART object มา
        :param i2c: ถ้าใช้ I2C ให้ส่ง I2C object มา
        :param i2c_addr: I2C address (default: 0x19)
        """
        self._uart = uart
        self._i2c = i2c
        self._i2c_addr = i2c_addr
        
        self._pm1_0 = 0.0
        self._pm2_5 = 0.0
        self._pm10 = 0.0
        
        # ตรวจสอบว่ามีการให้ interface มาอย่างน้อย 1 อย่าง
        if not uart and not i2c:
            raise ValueError("Either UART or I2C must be provided")
            
        # เตรียมเซ็นเซอร์ให้พร้อมใช้งาน
        if self._i2c:
            # I2C initialization if needed
            pass
        elif self._uart:
            # UART initialization if needed
            pass
            
    @property
    def pm1_0(self):
        """Return the PM1.0 concentration, in µg/m^3"""
        return self._pm1_0
        
    @property
    def pm2_5(self):
        """Return the PM2.5 concentration, in µg/m^3"""
        return self._pm2_5
    
    @property
    def pm10(self):
        """Return the PM10 concentration, in µg/m^3"""
        return self._pm10
    
    def read(self):
        """
        Read sensor data and update the PM values
        
        :return: True if reading was successful, False otherwise
        """
        if self._i2c:
            return self._read_i2c()
        elif self._uart:
            return self._read_uart()
        return False
            
    def _read_i2c(self):
        """Read sensor data via I2C"""
        try:
            # ส่งคำสั่งอ่านข้อมูล
            self._i2c.writeto(self._i2c_addr, bytes([self.CMD_READ_PARTICLE]))
            utime.sleep_ms(20)  # รอให้เซ็นเซอร์ประมวลผล
            
            # อ่านข้อมูล (รูปแบบขึ้นอยู่กับเซ็นเซอร์ที่ใช้)
            data = self._i2c.readfrom(self._i2c_addr, 6)
            
            # แปลงข้อมูลที่อ่านได้เป็นค่า PM
            self._pm1_0 = (data[0] << 8 | data[1]) / 10.0
            self._pm2_5 = (data[2] << 8 | data[3]) / 10.0
            self._pm10 = (data[4] << 8 | data[5]) / 10.0
            
            return True
            
        except Exception as e:
            print("I2C read error:", e)
            return False
    
    def _read_uart(self):
        """Read sensor data via UART"""
        try:
            # อ่านข้อมูลจาก UART
            # รูปแบบข้อมูลอาจแตกต่างกันไปตามรุ่นของเซ็นเซอร์
            
            # ล้าง buffer ก่อนอ่านข้อมูลใหม่
            while self._uart.any():
                self._uart.read(1)
                
            # ส่งคำสั่งอ่านข้อมูล (ถ้าจำเป็น)
            self._uart.write(bytes([0xFF, 0x01, self.CMD_READ_PARTICLE, 0x00, 0x00, 0x00, 0x00, 0x00]))
            
            utime.sleep_ms(100)  # รอให้เซ็นเซอร์ตอบกลับ
            
            if self._uart.any() < 10:  # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
                return False
                
            # อ่านข้อมูลและหาจุดเริ่มต้นของแพ็คเก็ต
            buffer = self._uart.read(32)  # อ่านข้อมูลให้มากเกินพอ
            
            # ค้นหา header ของแพ็คเก็ต (0x42, 0x4D)
            idx = -1
            for i in range(len(buffer) - 1):
                if buffer[i] == 0x42 and buffer[i+1] == 0x4D:
                    idx = i
                    break
                    
            if idx == -1:
                return False
                
            # แปลงข้อมูลเป็นค่า PM (รูปแบบขึ้นอยู่กับโปรโตคอลของเซ็นเซอร์)
            self._pm1_0 = (buffer[idx+4] << 8 | buffer[idx+5])
            self._pm2_5 = (buffer[idx+6] << 8 | buffer[idx+7])
            self._pm10 = (buffer[idx+8] << 8 | buffer[idx+9])
            
            return True
            
        except Exception as e:
            print("UART read error:", e)
            return False
    
    def sleep(self):
        """Put the sensor to sleep mode to save power"""
        # ส่งคำสั่ง sleep mode ถ้าเซ็นเซอร์รองรับ
        try:
            if self._i2c:
                self._i2c.writeto(self._i2c_addr, bytes([0x01]))
            elif self._uart:
                self._uart.write(bytes([0xFF, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]))
        except:
            pass
    
    def wake(self):
        """Wake up the sensor from sleep mode"""
        # ส่งคำสั่ง wake up ถ้าเซ็นเซอร์รองรับ
        try:
            if self._i2c:
                self._i2c.writeto(self._i2c_addr, bytes([0x02]))
            elif self._uart:
                self._uart.write(bytes([0xFF, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]))
        except:
            pass
