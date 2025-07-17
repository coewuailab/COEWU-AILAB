'use client'
// src/app/page.js
import { useState } from 'react';
import { Header, MonitoringPanel, MapSection, HistoryData, Footer } from '../components/MonitoringInterface';
import { ClientWrapper } from '../components/ClientWrapper';

export default function Home() {
  const [selectedLocation, setSelectedLocation] = useState(null);

  const handleLocationSelect = (locationData) => {
    console.log('Location selected in page.js:', locationData);
    setSelectedLocation(locationData);
  };

  const handleLocationClear = () => {
    setSelectedLocation(null);
  };

  return (
    <div className="bg-gray-50">
      <Header />

      <main className="flex flex-col lg:flex-row">
        {/* Left Panel - ให้มีขนาดที่เหมาะสม */}
        <div className="w-full lg:w-3/5 bg-white border-r border-gray-200">
          <div className="p-3">
            <MonitoringPanel
              selectedLocation={selectedLocation}
              onLocationClear={handleLocationClear}
            />
          </div>
        </div>

        {/* Right Panel - จัดเป็น column แนวตั้ง */}
        <div className="w-full lg:w-3/5 flex flex-col">
          {/* Map Section - ให้มีความสูงที่เหมาะสม */}
          <div className="bg-white border-b border-gray-200 h-[450px]">
            <MapSection
              selectedLocation={selectedLocation}
              onLocationSelect={handleLocationSelect}
            />
          </div>

          {/* History Data - ให้ขยายตามเนื้อหา */}
          <div className="bg-white">
            <div className="p-3">
              <HistoryData selectedLocation={selectedLocation} />
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}