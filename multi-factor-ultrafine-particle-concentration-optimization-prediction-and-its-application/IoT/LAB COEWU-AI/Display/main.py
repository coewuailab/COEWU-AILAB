import time
import math
from machine import Pin, SPI
import st7796s as st

# ST7796S Display Configuration
SPI_SCK = 18     # Clock pin connected to IO18 (SCK)
SPI_MOSI = 23    # Data pin connected to IO23 (SDI/MOSI)
SPI_MISO = 19    # MISO pin connected to IO19 (SDO/MISO)
ST7796S_CS = 5   # Chip select pin connected to IO5 (LCD_CS)
ST7796S_DC = 17  # Data/Command pin connected to IO17 (LCD_RS)
ST7796S_RST = 4  # Reset pin connected to IO4 
ST7796S_BL = 16  # Backlight pin connected to IO16 (LED pin on display)

# Colors (RGB565 format)
WHITE = 0xFFFF
BLACK = 0x0000
RED = 0xF800
GREEN = 0x07E0
BLUE = 0x001F
YELLOW = 0xFFE0
ORANGE = 0xFD20
DARK_GRAY = 0x2104
LIGHT_GRAY = 0xA514

# Initialize SPI
spi = SPI(2, baudrate=10000000, polarity=0, phase=0, sck=Pin(SPI_SCK), mosi=Pin(SPI_MOSI), miso=Pin(SPI_MISO))

# Initialize display in landscape mode (rotation=1)
print("Initializing display...")
tft = st.ST7796S(spi, 
                cs=Pin(ST7796S_CS, Pin.OUT), 
                dc=Pin(ST7796S_DC, Pin.OUT), 
                rst=Pin(ST7796S_RST, Pin.OUT), 
                bl=Pin(ST7796S_BL, Pin.OUT), 
                width=320,  
                height=480, 
                rotation=3)  # Landscape mode

# Turn on backlight
tft.backlight(1)
time.sleep(0.5)

# Clear display
tft.fill(DARK_GRAY)
time.sleep(0.1)

# Helper functions for drawing lines
def hline(x, y, length, color):
    """Draw horizontal line"""
    tft.fill_rect(x, y, length, 2, color)

def vline(x, y, length, color):
    """Draw vertical line"""
    tft.fill_rect(x, y, 2, length, color)

def draw_text(text, x, y, color, size=1):
    """Draw text at position"""
    tft.text(text, x, y, color, size)

def draw_large_text(text, x, y, color):
    """Draw large text"""
    tft.text(text, x, y, color, 4)  # Use size=4 for large text

def draw_color_scale(x, y, width, height):
    """Draw the air quality color scale"""
    colors = [BLUE, GREEN, YELLOW, ORANGE, RED]
    labels = ["Very good", "Good", "Moderate", "Health Effect", "Hazardous"]
    
    section_width = width // 5
    
    for i, color in enumerate(colors):
        section_x = x + i * section_width
        tft.fill_rect(section_x, y, section_width-1, height, color)
        time.sleep(0.01)
        
        # Draw labels below
        if i == 0:
            draw_text("Very", section_x+5, y+height+5, WHITE, 1)
            draw_text("good", section_x+5, y+height+15, WHITE, 1)
        elif i == 3:
            draw_text("Health", section_x+5, y+height+5, WHITE, 1)
            draw_text("Effect", section_x+5, y+height+15, WHITE, 1)
        else:
            draw_text(labels[i], section_x+5, y+height+5, WHITE)

def create_landscape_layout():
    """Create the main layout for landscape mode"""
    print("Creating landscape layout...")
    
    # Clear display
    tft.fill(DARK_GRAY)
    time.sleep(0.1)
    
    # Draw "Most Pick" header
    draw_text("UFPs Monitoring", 10, 10, LIGHT_GRAY, 2)
    
    # Main white container - adjusted for landscape
    container_x, container_y = 10, 40
    container_width, container_height = 460, 260
    tft.fill_rect(container_x, container_y, container_width, container_height, WHITE)
    time.sleep(0.1)
    
    # Date and time header
    draw_text("2 May 2025", container_x + 20, container_y + 10, BLACK, 2)
    draw_text("14:31:11", container_x + 320, container_y + 10, BLACK, 2)
    
    # Draw divider line
    hline(container_x, container_y + 35, container_width, BLACK)
    time.sleep(0.01)
    
    # Left section - PM0.1
    draw_text("PM0.1", container_x + 20, container_y + 50, BLACK, 2)
    draw_text("(ug/m3)", container_x + 120, container_y + 50, BLACK)
    
    # Draw large PM0.1 value
    draw_large_text("15.67", container_x + 60, container_y + 120, RED)
    
    # Color scale
    draw_color_scale(container_x + 10, container_y + 180, 240, 20)
    
    # Draw vertical divider
    divider_x = container_x + 270
    vline(divider_x - 10, container_y + 35, container_height - 35, BLACK)
    time.sleep(0.01)
    
    # Right section - Other measurements
    right_x = divider_x + 15
    spacing_y = 40
    
    # PM2.5
    draw_text("PM2.5", right_x - 20, container_y + 50, BLACK, 2)
    draw_text("(ug/m3)", right_x + 80, container_y + 50, BLACK)
    draw_text("17.26", right_x + 20, container_y + 70, GREEN, 2)
    
    # PM10
    draw_text("PM10", right_x - 20, container_y + 50 + spacing_y, BLACK, 2)
    draw_text("(ug/m3)", right_x + 80, container_y + 50 + spacing_y, BLACK)
    draw_text("38.28", right_x + 20, container_y + 70 + spacing_y, YELLOW, 2)
    
    # Humidity
    draw_text("Humidity", right_x - 20, container_y + 50 + spacing_y*2, BLACK, 2)
    draw_text("98 %", right_x + 20, container_y + 70 + spacing_y*2, BLACK, 2)
    
    # Temperature
    draw_text("Temperature", right_x - 20, container_y + 50 + spacing_y*3, BLACK, 2)
    draw_text("120 C", right_x + 20, container_y + 70 + spacing_y*3, BLACK, 2)

# Function to update values from sensors
def update_sensor_values(pm01=15.67, pm25=17.26, pm10=38.28, humidity=98, temperature=120):
    """Update display with new sensor values"""
    # Adjusted coordinates for landscape mode
    # Clear value areas before updating
    tft.fill_rect(50, 130, 200, 40, WHITE)  # PM0.1 area
    tft.fill_rect(325, 110, 120, 20, WHITE)  # PM2.5 area
    tft.fill_rect(325, 150, 120, 20, WHITE)  # PM10 area
    tft.fill_rect(325, 190, 120, 20, WHITE)  # Humidity area
    tft.fill_rect(325, 230, 120, 20, WHITE)  # Temperature area
    
    # Update values
    draw_large_text(f"{pm01:.2f}", 50, 130, RED)
    draw_text(f"{pm25:.2f}", 325, 110, GREEN, 2)
    draw_text(f"{pm10:.2f}", 325, 150, YELLOW, 2)
    draw_text(f"{humidity} %", 325, 190, BLACK, 2)
    draw_text(f"{temperature} C", 325, 230, BLACK, 2)

# Test display with full screen
print("Testing display...")
tft.fill(BLACK)
time.sleep(0.5)

# Create the landscape layout
create_landscape_layout()
print("Layout created successfully!")

# Debug information
print(f"Display dimensions: {tft.width}x{tft.height}")

# Main loop
while True:
    # Example of updating values
    # update_sensor_values(pm01=15.67, pm25=17.26, pm10=38.28, humidity=98, temperature=120)
    time.sleep(5)