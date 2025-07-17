
// src/app/layout.js (แก้ไขแล้ว)
import './globals.css'
import 'leaflet/dist/leaflet.css'

export const metadata = {
  title: 'Interactive Map',
  description: 'Interactive map with Leaflet',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css" />
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css" />
        {/* Material Symbols - ใช้แบบรวมทั้งชุด */}
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
        {/* หรือใช้แบบเก่า Material Icons */}
        <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons" />
      </head>
      <body>{children}</body>
    </html>
  )
}