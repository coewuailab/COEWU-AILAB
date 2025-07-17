// src/components/ClientWrapper.js - Fixed version
'use client'

import { useEffect, useState } from 'react'
import dynamic from 'next/dynamic'

// Dynamically import MapComponent to avoid SSR issues
// แก้ไข path ให้ถูกต้อง - ใช้ MapComponent แทน MapComponents
const MapComponent = dynamic(() => import('./MapComponents'), {
  ssr: false,
  loading: () => (
    <div className="absolute inset-0 flex items-center justify-center bg-gray-100">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4 mx-auto"></div>
        <p className="text-lg font-sarabun text-gray-600">กำลังโหลดแผนที่...</p>
        <p className="text-sm font-sarabun text-gray-500 mt-2">กรุณารอสักครู่</p>
      </div>
    </div>
  )
})

export function ClientWrapper({ onLocationSelect }) {
  const [isClient, setIsClient] = useState(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  const handleLocationSelect = (locationData) => {
    console.log('=== CLIENT WRAPPER DEBUG ===');
    console.log('Location selected in ClientWrapper:', locationData);
    console.log('Location ID:', locationData?.id);
    console.log('Location Name:', locationData?.name);
    console.log('onLocationSelect function exists:', !!onLocationSelect);
    console.log('============================');

    if (onLocationSelect && typeof onLocationSelect === 'function') {
      onLocationSelect(locationData);
    }
  };

  if (!isClient) {
    return (
      <div className="absolute inset-0 flex items-center justify-center bg-gray-100">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4 mx-auto"></div>
          <p className="text-lg font-sarabun text-gray-600">กำลังเตรียมแผนที่...</p>
          <p className="text-sm font-sarabun text-gray-500 mt-2">กำลังตรวจสอบสภาพแวดล้อม</p>
        </div>
      </div>
    )
  }

  return (
    <div className="absolute inset-0">
      <MapComponent onLocationSelect={handleLocationSelect} />
    </div>
  )
}