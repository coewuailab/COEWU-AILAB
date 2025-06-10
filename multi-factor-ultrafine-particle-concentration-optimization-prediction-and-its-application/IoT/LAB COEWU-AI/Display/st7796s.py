"""
MicroPython ST7796U/ST7796S TFT display driver
"""

import time
from micropython import const
import framebuf

# Register definitions
_SWRESET = const(0x01)
_SLPIN = const(0x10)
_SLPOUT = const(0x11)
_NORON = const(0x13)
_INVOFF = const(0x20)
_INVON = const(0x21)
_DISPON = const(0x29)
_CASET = const(0x2A)
_RASET = const(0x2B)
_RAMWR = const(0x2C)
_MADCTL = const(0x36)
_COLMOD = const(0x3A)

# Color definitions
BLACK = const(0x0000)
BLUE = const(0x001F)
RED = const(0xF800)
GREEN = const(0x07E0)
CYAN = const(0x07FF)
MAGENTA = const(0xF81F)
YELLOW = const(0xFFE0)
WHITE = const(0xFFFF)

_BUFFER_SIZE = const(512)

def color565(r, g, b):
    """Convert r, g, b in range 0-255 to a 16-bit color value."""
    return (r & 0xF8) << 8 | (g & 0xFC) << 3 | b >> 3

class ST7796S:
    """ST7796S/ST7796U TFT display driver."""
    
    def __init__(self, spi, cs, dc, rst=None, bl=None, width=320, height=480, rotation=0):
        """Initialize the display."""
        self.spi = spi
        self.cs = cs
        self.dc = dc
        self.rst = rst
        self.bl = bl
        self.rotation = rotation
        
        # Store default dimensions
        self._width = width
        self._height = height
        
        # Set actual dimensions based on rotation
        if rotation in [0, 2]:  # Portrait
            self.width = width
            self.height = height
        else:  # Landscape
            self.width = height
            self.height = width
        
        # Initialize pins
        self.cs.init(self.cs.OUT, value=1)
        self.dc.init(self.dc.OUT, value=0)
        if self.rst:
            self.rst.init(self.rst.OUT, value=1)
        if self.bl:
            self.bl.init(self.bl.OUT, value=1)
            
        self.buffer = bytearray(_BUFFER_SIZE * 2)
        
        self.reset()
        self._init()
        self.set_rotation(rotation)
        self.fill(0)

    def reset(self):
        """Hard reset the display."""
        if self.rst:
            self.rst.value(1)
            time.sleep_ms(5)
            self.rst.value(0)
            time.sleep_ms(20)
            self.rst.value(1)
            time.sleep_ms(150)
        else:
            self._write_cmd(_SWRESET)
            time.sleep_ms(150)

    def _init(self):
        """Initialize the display."""
        # Exit sleep mode
        self._write_cmd(_SLPOUT)
        time.sleep_ms(120)
        
        # Interface Pixel Format
        self._write_cmd(_COLMOD)
        self._write_data(bytearray([0x55]))  # 16-bit color
        
        # Display inversion off
        self._write_cmd(_INVOFF)
        
        # Normal display on
        self._write_cmd(_NORON)
        time.sleep_ms(10)
        
        # Display on
        self._write_cmd(_DISPON)
        time.sleep_ms(100)

    def _write_cmd(self, cmd):
        """Write a command to the display."""
        self.dc.value(0)
        self.cs.value(0)
        self.spi.write(bytearray([cmd]))
        self.cs.value(1)

    def _write_data(self, data):
        """Write data to the display."""
        self.dc.value(1)
        self.cs.value(0)
        self.spi.write(data)
        self.cs.value(1)

    def set_rotation(self, rotation):
        """Set display rotation."""
        self.rotation = rotation
        
        # Adjust dimensions based on rotation
        if rotation in [0, 2]:  # Portrait
            self.width = self._width
            self.height = self._height
        else:  # Landscape
            self.width = self._height
            self.height = self._width
        
        # MADCTL register values for different rotations
        if rotation == 0:  # Portrait
            madctl = 0x48  # MX=1, MY=0, MV=0, ML=0, BGR=1
        elif rotation == 1:  # Landscape (90 degrees)
            madctl = 0x28  # MX=0, MY=0, MV=1, ML=0, BGR=1
        elif rotation == 2:  # Inverted Portrait (180 degrees)
            madctl = 0x88  # MX=0, MY=1, MV=0, ML=0, BGR=1
        elif rotation == 3:  # Inverted Landscape (270 degrees)
            madctl = 0xE8  # MX=1, MY=1, MV=1, ML=0, BGR=1
        else:
            madctl = 0x48  # Default
            
        self._write_cmd(_MADCTL)
        self._write_data(bytearray([madctl]))

    def _set_window(self, x0, y0, x1, y1):
        """Set the drawing window."""
        # Column address set
        self._write_cmd(_CASET)
        self._write_data(bytearray([x0 >> 8, x0 & 0xFF, x1 >> 8, x1 & 0xFF]))
        
        # Row address set
        self._write_cmd(_RASET)
        self._write_data(bytearray([y0 >> 8, y0 & 0xFF, y1 >> 8, y1 & 0xFF]))
        
        # Write to RAM
        self._write_cmd(_RAMWR)

    def fill(self, color):
        """Fill the display with a color."""
        self._set_window(0, 0, self.width - 1, self.height - 1)
        
        color_hi = color >> 8
        color_lo = color & 0xFF
        
        # Fill buffer with color
        for i in range(0, _BUFFER_SIZE * 2, 2):
            self.buffer[i] = color_hi
            self.buffer[i + 1] = color_lo
        
        # Calculate total pixels
        total_pixels = self.width * self.height
        pixels_per_chunk = _BUFFER_SIZE
        chunks = total_pixels // pixels_per_chunk
        remainder = total_pixels % pixels_per_chunk
        
        self.dc.value(1)
        self.cs.value(0)
        
        # Send chunks
        for _ in range(chunks):
            self.spi.write(self.buffer)
            
        # Send remainder
        if remainder:
            self.spi.write(self.buffer[:remainder * 2])
            
        self.cs.value(1)

    def fill_rect(self, x, y, w, h, color):
        """Draw a filled rectangle."""
        # Clip to display boundaries
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        if x + w > self.width:
            w = self.width - x
        if y + h > self.height:
            h = self.height - y
        
        if w <= 0 or h <= 0:
            return
            
        self._set_window(x, y, x + w - 1, y + h - 1)
        
        color_hi = color >> 8
        color_lo = color & 0xFF
        
        # Fill buffer with color
        for i in range(0, _BUFFER_SIZE * 2, 2):
            self.buffer[i] = color_hi
            self.buffer[i + 1] = color_lo
        
        # Calculate total pixels
        total_pixels = w * h
        pixels_per_chunk = _BUFFER_SIZE
        chunks = total_pixels // pixels_per_chunk
        remainder = total_pixels % pixels_per_chunk
        
        self.dc.value(1)
        self.cs.value(0)
        
        # Send chunks
        for _ in range(chunks):
            self.spi.write(self.buffer)
            
        # Send remainder
        if remainder:
            self.spi.write(self.buffer[:remainder * 2])
            
        self.cs.value(1)

    def pixel(self, x, y, color):
        """Draw a pixel."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self._set_window(x, y, x, y)
            self._write_data(bytearray([color >> 8, color & 0xFF]))

    def text(self, text, x, y, color, size=1):
        """Draw text on the display using basic font."""
        import framebuf
        
        # Create a small framebuffer for a single character
        char_width = 8
        char_height = 8
        char_buf = bytearray(char_width)
        char_fb = framebuf.FrameBuffer(char_buf, char_width, char_height, framebuf.MONO_HLSB)
        
        for i, char in enumerate(text):
            char_x = x + i * char_width * size
            
            # Skip if character is completely off screen
            if char_x >= self.width or char_x + char_width * size < 0:
                continue
                
            # Clear the character buffer
            for j in range(len(char_buf)):
                char_buf[j] = 0
                
            # Draw the character into the buffer
            char_fb.text(char, 0, 0, 1)
            
            # Render the character to the display
            for row in range(char_height):
                for col in range(char_width):
                    if char_fb.pixel(col, row):
                        # Draw a rectangle for each pixel if size > 1
                        px = char_x + col * size
                        py = y + row * size
                        
                        # Check if pixel is within display bounds
                        if 0 <= px < self.width and 0 <= py < self.height:
                            if size == 1:
                                self.pixel(px, py, color)
                            else:
                                self.fill_rect(px, py, size, size, color)

    def show(self):
        """Update the display. For compatibility with other display drivers."""
        pass
        
    def backlight(self, state):
        """Control the backlight."""
        if self.bl:
            self.bl.value(1 if state else 0)