'use client';

import React, { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import Link from 'next/link';
import { getDatabase, ref, get, query, orderByKey, limitToLast } from 'firebase/database';
import { initializeApp, getApps } from 'firebase/app';
import { LOCATION_CONFIGS } from '../config/firebase-configs';
import { ArrowUpRight, Thermometer, Droplets, Menu, X, ChevronUp, MapPin, Clock, TrendingUp, BarChart3, UserX, Users, BookOpen, Cloud } from 'lucide-react';
import {
  getAirQualityColor,
  formatPMValue,
  PM_THRESHOLDS,
  getRecommendationIcon,
  useMonitoringData,
  useLocationMonitoringData,
  getGeneralRecommendations,
  getSensitiveRecommendations,
  determineAirQuality,
  getPMDetails,
  getPMReadingStatusAndColor
} from '../data/monitoring-data';
import { LiveActivityNumber, useReducedMotion } from './Animation';
import { TriviaPopupContent } from '../data/trivia';
import './MapComponents';
import HistoryData from './HistoryData';

// Header Component
const Header = ({ selectedLocation }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 h-18 sm:h-20 bg-gradient-to-r from-green-100 to-green-300 border-b-2 border-gray-200 flex-shrink-0">
      <div className="h-full flex items-center justify-between px-4 sm:px-6">
        <div className="flex items-center gap-4 sm:gap-6">
          <div className="w-20 h-20 sm:w-24 sm:h-24 flex-shrink-0 flex items-center justify-center">
            <img
              src="/assets/images/logo.png"
              alt="Mupcop logo"
              className="w-20 h-20 sm:w-24 sm:h-24 object-contain"
              width={96}
              height={96}
            />
          </div>
          <div>
            <h1 className="text-xl sm:text-2xl lg:text-3xl xl:text-4xl font-bold text-black font-montserrat">VISTA</h1>
            <h2 className="text-base sm:text-lg lg:text-xl font-light text-black max-w-xl font-montserrat">
              Vulnerable Indoor Sensitive Tiny Aerosol monitor
            </h2>
          </div>
        </div>

        <nav className="hidden sm:flex items-center gap-4 sm:gap-6">
          <Link href="/" className="px-4 py-2 text-lg sm:text-xl lg:text-2xl font-semibold text-black rounded-lg hover:bg-green-200 transition-colors font-montserrat">
            Air quality
          </Link>
          <Link href="/history" className="px-4 py-2 text-lg sm:text-xl lg:text-2xl font-semibold text-black rounded-lg hover:bg-green-200 transition-colors font-montserrat">
            History Data
          </Link>
        </nav>

        <button
          className="sm:hidden"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
          {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>
    </header>
  );
};

// MonitoringPanel Component
const MonitoringPanel = ({ selectedLocation, onLocationClear }) => {
  const [showRecommendationPopup, setShowRecommendationPopup] = useState(false);
  const [selectedRecommendations, setSelectedRecommendations] = useState([]);
  const [showPMDetailPopup, setShowPMDetailPopup] = useState(false);
  const [showSensitiveGroupPopup, setShowSensitiveGroupPopup] = useState(false);
  const [showGeneralGroupPopup, setShowGeneralGroupPopup] = useState(false);
  const [showTriviaPopup, setShowTriviaPopup] = useState(false);
  const [showPMDetailsPopup, setShowPMDetailsPopup] = useState(false);
  const [selectedPMType, setSelectedPMType] = useState(null);
  const [selectedLevel, setSelectedLevel] = useState(null);
  const [selectedGroupType, setSelectedGroupType] = useState('general');

  const handlePMReadingClick = (pmType) => {
    setSelectedPMType(pmType);
    setShowPMDetailPopup(true);
  };

  const closePMDetailPopup = () => {
    setShowPMDetailPopup(false);
    setSelectedPMType(null);
  };

  const handleSensitiveGroupClick = () => setShowSensitiveGroupPopup(true);
  const handleGeneralGroupClick = () => setShowGeneralGroupPopup(true);
  const handleTriviaClick = () => setShowTriviaPopup(true);
  const handlePMDetailsClick = () => setShowPMDetailsPopup(true);

  const closePopup = () => {
    setShowRecommendationPopup(false);
    setSelectedRecommendations([]);
  };

  const { data: defaultData, loading: defaultLoading, error: defaultError } = useMonitoringData();
  const { data: locationData, loading: locationLoading, error: locationError } = useLocationMonitoringData(selectedLocation);

  const data = selectedLocation ? locationData : defaultData;
  const loading = selectedLocation ? locationLoading : defaultLoading;
  const error = selectedLocation ? locationError : defaultError;

  const isDataValid = data && data.pmReadings && Array.isArray(data.pmReadings);
  const statusColor = isDataValid ? getAirQualityColor(data.mainReading?.status || 'Good') : '#2DC653';

  const getSafeValue = (readings, type) => {
    if (!readings || !Array.isArray(readings)) return 0;
    const reading = readings.find(r => r && r.type === type);
    return parseFloat(reading?.value || 0) || 0;
  };

  const calculateGroupStatus = (groupType = 'general') => {
    if (!isDataValid) return 'Good';

    const pc01Value = getSafeValue(data.pmReadings, 'PC0.1');
    const pm01Value = getSafeValue(data.pmReadings, 'PM0.1');
    const pm25Value = getSafeValue(data.pmReadings, 'PM2.5');
    const pm10Value = getSafeValue(data.pmReadings, 'PM10');

    return determineAirQuality(pc01Value, pm01Value, pm25Value, pm10Value);
  };

  const sensitiveStatus = calculateGroupStatus('sensitive');
  const generalStatus = calculateGroupStatus('general');

  const handleRecommendationClick = (groupType) => {
    const status = groupType === 'sensitive' ? sensitiveStatus : generalStatus;
    const recommendations = groupType === 'sensitive'
      ? getSensitiveRecommendations(status)
      : getGeneralRecommendations(status);

    setSelectedRecommendations(recommendations);
    setSelectedGroupType(groupType);
    setShowRecommendationPopup(true);
  };

  useEffect(() => {
    const handleEscapeKey = (event) => {
      if (event.key === 'Escape') {
        if (showRecommendationPopup) closePopup();
        if (showPMDetailPopup) closePMDetailPopup();
        if (showSensitiveGroupPopup) setShowSensitiveGroupPopup(false);
        if (showGeneralGroupPopup) setShowGeneralGroupPopup(false);
        if (showTriviaPopup) setShowTriviaPopup(false);
        if (showPMDetailsPopup) setShowPMDetailsPopup(false);
      }
    };

    if (showRecommendationPopup || showPMDetailPopup || showSensitiveGroupPopup || showGeneralGroupPopup || showTriviaPopup || showPMDetailsPopup) {
      document.addEventListener('keydown', handleEscapeKey);
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }

    return () => {
      document.removeEventListener('keydown', handleEscapeKey);
      document.body.style.overflow = 'unset';
    };
  }, [showRecommendationPopup, showPMDetailPopup, showSensitiveGroupPopup, showGeneralGroupPopup, showTriviaPopup, showPMDetailsPopup]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px] w-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-800 mb-3"></div>
          <p className="text-lg sm:text-xl text-gray-600 font-sarabun">กำลังโหลดข้อมูล...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px] w-full">
        <div className="text-center">
          <div className="text-red-500 mb-3">⚠️</div>
          <p className="text-lg sm:text-xl text-red-600 font-sarabun">เกิดข้อผิดพลาด: {error}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-3 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
          >
            รีเฟรช
          </button>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center min-h-[400px] w-full">
        <div className="text-center">
          <div className="text-gray-400 mb-3">📭</div>
          <p className="text-lg sm:text-xl text-gray-600 font-sarabun">ไม่มีข้อมูล</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full bg-white">
      <div className="p-3 sm:p-4">
        <div className="flex items-center justify-between mb-3 sm:mb-4">
          <h2 className="text-xl sm:text-2xl lg:text-3xl xl:text-4xl font-light text-black font-montserrat">UFPs Monitoring</h2>
        </div>
      </div>

      <div className="bg-white rounded-lg p-3 sm:p-4 shadow-sm border border-gray-100 mb-3 sm:mb-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
            <span className="text-lg sm:text-xl lg:text-2xl xl:text-3xl font-semibold text-black font-montserrat">LIVE</span>
          </div>
          <span className="text-base sm:text-lg lg:text-xl text-black font-numbers">{data.date || 'N/A'}</span>
        </div>

        <div className="text-base sm:text-lg lg:text-xl text-gray-600 mb-3 sm:mb-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 sm:gap-0">
            <div className="flex items-center gap-2">
              <MapPin className="w-4 h-4 sm:w-5 sm:h-5" />
              <span className="truncate text-base sm:text-lg lg:text-xl font-sarabun">
                {selectedLocation ? selectedLocation.name : data.location || "Cafe Amazon สาขา ST"}
              </span>
            </div>
            <div className="text-base sm:text-lg lg:text-xl text-gray-500 sm:ml-2">
              <span className="font-sarabun">อัพเดท: </span>
              <span className="font-numbers">{data.time || 'N/A'}</span>
            </div>
          </div>
        </div>

        <div className="flex rounded-lg mb-3 sm:mb-4 shadow-sm overflow-hidden min-h-[140px] sm:min-h-[160px]">
          <div
            className="w-[50%] sm:w-[45%] p-2 sm:p-3 flex flex-col justify-center relative"
            style={{ backgroundColor: statusColor }}
          >
            <div className="absolute inset-0 bg-black bg-opacity-30"></div>
            <div className="relative z-10 text-center">
              <div className="text-sm sm:text-base lg:text-lg xl:text-xl font-bold mb-2 text-white opacity-90 font-montserrat">
                {data.mainReading?.type || 'PC0.1'}
              </div>
              <div className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-semibold text-white mb-2 leading-tight" style={{ textShadow: '3px 3px 6px rgba(75, 64, 64, 0.8)' }}>
                <LiveActivityNumber
                  value={data.mainReading?.value || getSafeValue(data.pmReadings, 'PC0.1')}
                  type="cascade-slide"
                  decimals={0}
                  className="text-white"
                  showChangeIndicator={true}
                  minDigits={1}
                  direction="right-to-left"
                />
              </div>
              <div className="text-sm sm:text-base lg:text-lg xl:text-xl font-semibold text-white opacity-90 font-montserrat">
                {data.mainReading?.unit || 'PNC'}
              </div>
            </div>
          </div>

          <div
            className="w-[50%] sm:w-[55%] flex flex-col"
            style={{ backgroundColor: statusColor }}
          >
            <button
              onClick={handleSensitiveGroupClick}
              className="flex-1 flex items-center gap-2 sm:gap-3 p-2 sm:p-3 transition-all hover:bg-black hover:bg-opacity-10 border-b border-white/20 hover:shadow-lg"
              aria-label="ดูคำแนะนำสำหรับกลุ่มเปราะบาง"
            >
              <div className="flex-shrink-0 ml-2 sm:ml-4">
                <span
                  className="material-symbols-outlined text-white drop-shadow-lg"
                  style={{
                    fontSize: 'clamp(2.5rem, 5vw, 3.5rem)',
                    textShadow: '2px 2px 4px rgba(0, 0, 0, 0.5)'
                  }}
                >
                  shield_person
                </span>
              </div>
              <div className="flex-1 text-center min-w-0 px-4">
                <div className="text-sm sm:text-base lg:text-2xl font-semibold text-white opacity-90 mb-2 font-sarabun">
                  กลุ่มเปราะบาง
                </div>
                <div className="text-base sm:text-lg lg:text-2xl xl:text-4xl font-bold text-white truncate font-montserrat transform hover:scale-125 transition-transform duration-300 ease-in-out" style={{ textShadow: '2px 2px 4px rgba(0,0,0,0.8)' }}>
                  {sensitiveStatus}
                </div>
              </div>
              <div className="flex-shrink-0 text-white text-sm sm:text-base opacity-60">
                →
              </div>
            </button>
          </div>
        </div>

        <div className="flex justify-center gap-8 sm:gap-12 text-base sm:text-lg lg:text-xl mb-3 sm:mb-4 py-3 sm:py-4 bg-gray-50 rounded-lg">
          <div className="flex items-center gap-3 sm:gap-4">
            <img
              src="/assets/images/temperature.png"
              alt="Temperature"
              className="w-8 h-8 sm:w-10 sm:h-10"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'inline';
              }}
            />
            <Thermometer className="w-5 h-5 sm:w-6 sm:h-6 text-gray-500" style={{ display: 'none' }} />
            <LiveActivityNumber
              value={parseFloat(data.conditions?.temperature?.replace(/[^\d.-]/g, '')) || 25.5}
              type="cascade-slide"
              decimals={1}
              className="text-gray-700 font-medium text-base sm:text-lg lg:text-xl xl:text-2xl"
              showChangeIndicator={false}
            />
            <span className="text-gray-700 font-medium text-base sm:text-lg lg:text-xl xl:text-2xl font-montserrat">°C</span>
          </div>

          <div className="flex items-center gap-3 sm:gap-4">
            <img
              src="/assets/images/humidity.png"
              alt="Humidity"
              className="w-8 h-8 sm:w-10 sm:h-10"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'inline';
              }}
            />
            <Droplets className="w-5 h-5 sm:w-6 sm:h-6 text-gray-500" style={{ display: 'none' }} />
            <LiveActivityNumber
              value={parseFloat(data.conditions?.humidity?.replace(/[^\d.-]/g, '')) || 65}
              type="cascade-slide"
              decimals={0}
              className="text-gray-700 font-medium text-base sm:text-lg lg:text-xl xl:text-2xl"
              showChangeIndicator={false}
              direction="left-to-right"
            />
            <span className="text-gray-700 font-medium text-base sm:text-lg lg:text-xl xl:text-2xl font-montserrat">%</span>
          </div>
        </div>

        <div
          className="pt-2 sm:pt-3 mt-2 sm:mt-3"
          style={{
            borderTop: '1px solid #e5e5e5',
          }}
        >
          <div className="text-center">
            {!data.recommendations || data.recommendations.length === 0 ? (
              <div className="text-xs sm:text-sm lg:text-base text-gray-500 mt-3 font-sarabun">
                ไม่มีคำแนะนำ
              </div>
            ) : (
              <>
                <div className="flex justify-center items-center gap-3 sm:gap-4 p-1 flex-nowrap overflow-x-auto">
                  {data.recommendations.map((recommendation, index) => {
                    const iconPath = getRecommendationIcon(recommendation);
                    const isImageIcon = typeof iconPath === 'string' && iconPath.match(/\.(png|jpg|jpeg|svg|gif)$/i);

                    return (
                      <div key={index} className="flex flex-col items-center">
                        <div
                          className="p-1 rounded-lg"
                          style={{
                            background: '#f8f8f8',
                            boxShadow: 'inset 2px 2px 4px #e0e0e0, inset -2px -2px 4px #ffffff',
                          }}
                        >
                          {isImageIcon ? (
                            <img
                              src={iconPath || '/assets/images/fallback-icon.png'}
                              alt={recommendation.name || 'Recommendation icon'}
                              className="w-4 h-4 sm:w-6 sm:h-6 lg:w-7 lg:h-7"
                              onError={(e) => {
                                console.error(`Failed to load image: ${iconPath}`);
                                e.target.src = '/assets/images/fallback-icon.png';
                              }}
                            />
                          ) : (
                            <span className="text-sm sm:text-base lg:text-lg">{iconPath || 'N/A'}</span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="text-xs sm:text-sm lg:text-base text-gray-500 mt-3 font-sarabun">
                  คลิกที่ Bubble ด้านบนเพื่อดูคำแนะนำสำหรับแต่ละกลุ่ม
                </div>
              </>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg p-2 sm:p-3 shadow-sm border border-gray-100 mb-2 sm:mb-3">
          <h3 className="text-base sm:text-lg lg:text-xl xl:text-2xl font-medium text-gray-800 mb-2 sm:mb-3 font-sarabun">ข้อมูลมลพิษทางอากาศ</h3>
          <div className="grid grid-cols-2 gap-1 sm:gap-2">
            {isDataValid ? (
              (() => {
                const pm25Reading = data.pmReadings.find(r => r && r.type === 'PM2.5');
                const pm10Reading = data.pmReadings.find(r => r && r.type === 'PM10');

                console.log('=== PM SUMMARY CARD DEBUG ===');
                console.log('All PM Readings:', data.pmReadings);
                console.log('PM2.5 Reading:', pm25Reading);
                console.log('PM10 Reading:', pm10Reading);
                console.log('============================');

                const pmDataToShow = [pm25Reading, pm10Reading].filter(Boolean);

                return pmDataToShow.length > 0 ? pmDataToShow.map((reading, index) => {
                  let numericValue = 0;
                  if (typeof reading.value === 'number') {
                    numericValue = reading.value;
                  } else if (reading.value !== undefined) {
                    numericValue = parseFloat(String(reading.value).replace(/[^\d.-]/g, '')) || 0;
                  }

                  // Get status and color for PM2.5
                  const pm25StatusAndColor = pm25Reading
                    ? getPMReadingStatusAndColor({
                      type: 'PM2.5',
                      value: parseFloat(pm25Reading.value) || 0,
                      pm25Value: parseFloat(pm25Reading.value) || 0
                    })
                    : { status: 'Good', color: '#2DC653' };

                  // Get status for PM10 (if needed), but use PM2.5's color
                  const pm10StatusAndColor = pm10Reading && reading.type === 'PM10'
                    ? getPMReadingStatusAndColor({
                      type: 'PM10',
                      value: numericValue,
                      pm25Value: parseFloat(pm25Reading?.value) || 0
                    })
                    : pm25StatusAndColor;

                  // Force PM10 to use PM2.5's color
                  const individualColor = pm25StatusAndColor.color;
                  const individualStatus = reading.type === 'PM2.5' ? pm25StatusAndColor.status : pm10StatusAndColor.status;

                  console.log(`=== ${reading.type} INDIVIDUAL STATUS ===`);
                  console.log('Value:', numericValue);
                  console.log('Individual Status:', individualStatus);
                  console.log('Individual Color:', individualColor);
                  console.log('=====================================');

                  return (
                    <button
                      key={`${reading.type}-${selectedLocation?.id || 'default'}-${index}`}
                      className="rounded-lg p-2 sm:p-3 text-white text-center bg-opacity-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-300 hover:shadow-lg"
                      style={{ backgroundColor: individualColor }}
                      onClick={() => handlePMReadingClick(reading.type)}
                      aria-label={`ดูคำแนะนำสำหรับ ${reading.type}`}
                    >
                      <div className="flex items-center justify-center gap-2">
                        <Cloud className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
                        <div className="text-sm sm:text-base lg:text-lg font-medium mb-1 opacity-90 font-montserrat">
                          <strong>{reading.type}</strong>
                        </div>
                      </div>
                      <div className="text-lg sm:text-2xl lg:text-3xl font-bold transform hover:scale-125 transition-transform duration-300 ease-in-out" style={{ textShadow: '2px 2px 4px rgba(0,0,0,0.8)' }}>
                        <LiveActivityNumber
                          value={numericValue}
                          type="cascade-slide"
                          decimals={2}
                          className="text-white"
                          showChangeIndicator={false}
                        />
                      </div>
                      <div className="text-xs sm:text-sm lg:text-base opacity-90 font-montserrat">
                        <strong>μg/m³</strong>
                      </div>
                    </button>
                  );
                }) : (
                  <div className="col-span-2 text-center text-xs sm:text-sm text-gray-500 py-4 font-sarabun">
                    ไม่พบข้อมูล PM2.5 และ PM10
                  </div>
                );
              })()
            ) : (
              <div className="col-span-2 text-center text-xs sm:text-sm text-gray-500 py-4 font-sarabun">
                ไม่มีข้อมูลมลพิษในขณะนี้
              </div>
            )}
          </div>
        </div>

        <SimpleChevronBar />
        <TriviaCard onTriviaClick={handleTriviaClick} />
        {showPMDetailPopup && selectedPMType && createPortal(
          <div className="fixed inset-0 flex items-center justify-center" style={{ zIndex: 9999, backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
            <div className="bg-white rounded-lg max-w-2xl min-w-[20rem] w-full mx-4 p-8 max-h-[90vh] overflow-y-auto shadow-2xl" onClick={(e) => e.stopPropagation()}>
              <div className="flex justify-between items-center mb-6">
                <h3 id="pm-detail-modal-title" className="text-2xl font-semibold text-black font-sarabun">
                  คำแนะนำสำหรับคนทั่วไป
                </h3>
                <button onClick={closePMDetailPopup} className="text-gray-500 hover:text-gray-700 text-3xl font-bold transition-colors p-1 rounded-md hover:bg-gray-100">
                  ×
                </button>
              </div>
              <div className="space-y-6">
                <div className="space-y-4">
                  {isDataValid && data.pmReadings && (
                    (() => {
                      const reading = data.pmReadings.find(r => r && r.type === selectedPMType);
                      const status = reading ? getPMReadingStatusAndColor(reading).status : 'Good';
                      const recommendations = getGeneralRecommendations(status);
                      return recommendations.map((rec, index) => (
                        <div key={index} className="flex items-center gap-6 p-6 bg-gray-50 rounded-lg">
                          {typeof getRecommendationIcon(rec) === 'string' && getRecommendationIcon(rec).startsWith('/assets/images/') ? (
                            <img src={getRecommendationIcon(rec)} alt="" className="w-12 h-12" />
                          ) : (
                            <span className="text-2xl">{getRecommendationIcon(rec)}</span>
                          )}
                          <p className="text-gray-600 text-lg font-sarabun">{rec}</p>
                        </div>
                      ));
                    })()
                  )}
                </div>
              </div>
              <div className="mt-8">
                <button onClick={closePMDetailPopup} className="w-full bg-green-500 hover:bg-green-600 text-white py-4 px-6 rounded-lg transition-colors text-xl font-medium font-sarabun">
                  ปิด
                </button>
              </div>
            </div>
          </div>,
          document.body
        )}

        {showSensitiveGroupPopup && createPortal(
          <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
            <div className="bg-white rounded-lg max-w-2xl min-w-[20rem] w-full mx-4 p-8 max-h-[90vh] overflow-y-auto shadow-2xl" onClick={e => e.stopPropagation()}>
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-2xl font-semibold text-black font-sarabun">คำแนะนำสำหรับกลุ่มเปราะบาง</h3>
                <button
                  onClick={() => setShowSensitiveGroupPopup(false)}
                  className="text-black hover:text-gray-700 text-3xl"
                >
                  ×
                </button>
              </div>
              <div className="space-y-6">
                <div className="space-y-4">
                  {getSensitiveRecommendations(sensitiveStatus).map((rec, index) => {
                    const iconPath = getRecommendationIcon(rec);
                    return (
                      <div key={index} className="flex items-center gap-6 p-6 bg-gray-50 rounded-lg">
                        {typeof iconPath === 'string' && iconPath.startsWith('/assets/images/') ? (
                          <img src={iconPath} alt="" className="w-12 h-12" />
                        ) : (
                          <span className="text-2xl">{iconPath}</span>
                        )}
                        <p className="text-gray-600 text-lg font-sarabun">{rec}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
              <div className="mt-8">
                <button onClick={() => setShowSensitiveGroupPopup(false)} className="w-full bg-green-500 hover:bg-green-600 text-white py-4 px-6 rounded-lg transition-colors text-xl font-medium font-sarabun">
                  ปิด
                </button>
              </div>
            </div>
          </div>,
          document.body
        )}

        {showGeneralGroupPopup && createPortal(
          <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
            <div className="bg-white rounded-lg max-w-2xl min-w-[20rem] w-full mx-4 p-8 max-h-[90vh] overflow-y-auto shadow-2xl" onClick={e => e.stopPropagation()}>
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-2xl font-semibold text-black font-sarabun">คำแนะนำสำหรับคนทั่วไป</h3>
                <button
                  onClick={() => setShowGeneralGroupPopup(false)}
                  className="text-gray-500 hover:text-gray-700 text-3xl"
                >
                  ×
                </button>
              </div>
              <div className="space-y-6">
                <div className="space-y-4">
                  {getGeneralRecommendations(generalStatus).map((rec, index) => {
                    const iconPath = getRecommendationIcon(rec);
                    return (
                      <div key={index} className="flex items-center gap-6 p-6 bg-gray-50 rounded-lg">
                        {typeof iconPath === 'string' && iconPath.startsWith('/assets/images/') ? (
                          <img src={iconPath} alt="" className="w-12 h-12" />
                        ) : (
                          <span className="text-2xl">{iconPath}</span>
                        )}
                        <p className="text-gray-600 text-lg font-sarabun">{rec}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
              <div className="mt-8">
                <button onClick={() => setShowGeneralGroupPopup(false)} className="w-full bg-green-500 hover:bg-green-600 text-white py-4 px-6 rounded-lg transition-colors text-xl font-medium font-sarabun">
                  ปิด
                </button>
              </div>
            </div>
          </div>,
          document.body
        )}

        {showTriviaPopup && createPortal(
          <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50 transition-opacity duration-300 ease-in-out" style={{ opacity: showTriviaPopup ? 1 : 0 }}>
            <TriviaPopupContent onClose={() => setShowTriviaPopup(false)} />
          </div>,
          document.body
        )}

        {showPMDetailsPopup && createPortal(
          <PMDetailsPopupContent onClose={() => setShowPMDetailsPopup(false)} />,
          document.body
        )}
      </div>
    </div>
  );
};

// MapSection Component
const MapSection = ({ selectedLocation, onLocationSelect }) => {
  return (
    <div className="flex-1 p-4 h-full bg-white">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base sm:text-lg lg:text-xl xl:text-2xl font-medium text-gray-800 mb-2 sm:mb-3 font-sarabun">แผนที่ตำแหน่งตรวจวัด</h2>
        <div className="text-sm text-gray-500 font-sarabun">
          คลิกที่หมุดเพื่อดูข้อมูล
        </div>
      </div>

      <div className="w-full h-[calc(100%-60px)] bg-gray-100 rounded-lg border border-gray-200 overflow-hidden">
        <MapComponentWrapper onLocationSelect={onLocationSelect} />
      </div>
    </div>
  );
};

// MapComponentWrapper Component
const MapComponentWrapper = ({ onLocationSelect }) => {
  const [MapComponents, setMapComponents] = useState(null);

  useEffect(() => {
    import('./MapComponents').then((module) => {
      setMapComponents(() => module.default);
    });
  }, []);

  if (!MapComponents) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-center text-gray-500">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-600 mx-auto mb-2"></div>
          <p className="text-sm font-sarabun">กำลังโหลดแผนที่...</p>
        </div>
      </div>
    );
  }

  return <MapComponents onLocationSelect={onLocationSelect} />;
};

// Footer Component
const Footer = () => (
  <footer className="h-18 sm:h-20 bg-gradient-to-r from-green-100 to-green-300 border-b-2 border-gray-200 flex-shrink-0 flex items-center justify-between p-4 sm:p-6">
    <span className="text-base sm:text-lg lg:text-xl xl:text-2xl text-black font-montserrat">© 2025 Jaejae Dream Yok. All rights reserved.</span>
    <div className="flex items-center gap-4 sm:gap-6">
      <div className="flex-shrink-0 flex items-center justify-center">
        <img
          src="/assets/images/logo.png"
          alt="Mupcop logo"
          className="w-20 h-20 sm:w-24 sm:h-24 object-contain"
          width={96}
          height={96}
        />
      </div>
      <div>
        <p className="text-xl sm:text-2xl lg:text-3xl xl:text-4xl font-bold text-black font-montserrat">VISTA</p>
      </div>
    </div>
  </footer>
);

// SimpleChevronBar Component
const SimpleChevronBar = () => {
  const levelKeys = Object.keys(PM_THRESHOLDS?.PC01 || {});
  const levels = levelKeys.map((level, index) => {
    const prevThresholds = index > 0 ? {
      pc01: PM_THRESHOLDS?.PC01?.[levelKeys[index - 1]] || 0,
      pm25: PM_THRESHOLDS?.PM?.[levelKeys[index - 1]] || 0, // Corrected typo from 'PM' to 'PM2.5'
      pm10: PM_THRESHOLDS?.PM10?.[levelKeys[index - 1]] || 0,
    } : { pc01: 0, pm25: 0, pm10: 0 };

    return {
      label: level,
      color: getAirQualityColor(level),
      pc01: PM_THRESHOLDS?.PC01?.[level] || 0,
      pm25: PM_THRESHOLDS?.PM?.[level] || 0, // Corrected typo
      pm10: PM_THRESHOLDS?.PM10?.[level] || 0,
      ranges: {
        pc01: {
          min: prevThresholds.pc01,
          max: PM_THRESHOLDS?.PC01?.[level] || Infinity,
        },
        pm25: {
          min: prevThresholds.pm25,
          max: PM_THRESHOLDS?.PM?.[level] || Infinity, // Corrected typo
        },
        pm10: {
          min: prevThresholds.pm10,
          max: PM_THRESHOLDS?.PM10?.[level] || Infinity,
        },
      },
    };
  });

  const [selectedLevel, setSelectedLevel] = useState(null);

  const handleLevelClick = (level) => {
    setSelectedLevel(level);
  };

  const closePopup = () => {
    setSelectedLevel(null);
  };

  const chevronWidth = 130;
  const chevronHeight = 50;
  const svgWidth = chevronWidth * levels.length;

  const formatRange = (min, max, unit = '') => {
    if (max === Infinity) {
      return `> ${min} ${unit}`;
    }
    if (min === 0) {
      return `≤ ${max} ${unit}`;
    }
    return `> ${min} และ ≤ ${max} ${unit}`;
  };

  return (
    <div className="bg-white rounded-lg p-3 sm:p-4 shadow-sm border border-gray-100 mb-3 sm:mb-4">
      <h3 className="text-base sm:text-lg lg:text-2xl font-medium text-gray-800 mb-3 font-sarabun">
        ระดับคุณภาพอากาศ
      </h3>
      <div className="relative mb-3">
        <svg viewBox={`0 0 ${svgWidth} ${chevronHeight}`} className="w-full h-12 sm:h-14">
          {levels.map((level, index) => {
            const x = index * chevronWidth;
            return (
              <g key={level.label} className="hover:shadow-xl hover:-translate-y-1 transition-all duration-300 ease-in-out">
                <path
                  d={`M ${x} 5 
                    L ${x + chevronWidth - 10} 5 
                    L ${x + chevronWidth} 25 
                    L ${x + chevronWidth - 10} 45 
                    L ${x} 45 
                    ${index > 0 ? `L ${x + 10} 25` : `L ${x} 25`} 
                    Z`}
                  fill={level.color}
                  stroke="rgba(255,255,255,0.5)"
                  strokeWidth="1"
                  onClick={() => handleLevelClick(level)}
                  style={{ cursor: 'pointer' }}
                />
                <text
                  x={x + chevronWidth / 2}
                  y="28"
                  textAnchor="middle"
                  fill="white"
                  fontSize="15"
                  fontWeight="600"
                  className="font-montserrat"
                  style={{ textShadow: '1px 1px 1px rgba(0,0,0,0.8)' }}
                >
                  {level.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {selectedLevel && createPortal(
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-white rounded-lg max-w-2xl min-w-[20rem] w-full mx-4 p-8 max-h-[90vh] overflow-y-auto shadow-2xl" onClick={(e) => e.stopPropagation()} style={{ maxHeight: '80vh', overflowY: 'auto' }}>
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-2xl font-semibold text-black font-sarabun">{selectedLevel.label}</h3>
              <button
                onClick={closePopup}
                className="text-gray-500 hover:text-gray-700 text-3xl"
              >
                ×
              </button>
            </div>
            <div className="space-y-6">
              <div className="text-lg text-gray-600 font-sarabun">
                <strong>PC0.1:</strong> {formatRange(selectedLevel.ranges.pc01.min, selectedLevel.ranges.pc01.max, 'PNC')}
              </div>
              <div className="text-lg text-gray-600 font-sarabun">
                <strong>PM2.5:</strong> {formatRange(selectedLevel.ranges.pm25.min, selectedLevel.ranges.pm25.max, 'μg/m³')}
              </div>
              <div className="text-lg text-gray-600 font-sarabun">
                <strong>PM10:</strong> {formatRange(selectedLevel.ranges.pm10.min, selectedLevel.ranges.pm10.max, 'μg/m³')}
              </div>
            </div>
            <div className="mt-8">
              <button onClick={closePopup} className="w-full bg-green-500 hover:bg-green-600 text-white py-4 px-6 rounded-lg transition-colors text-xl font-medium font-sarabun">
                ปิด
              </button>
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
};

// TriviaCard Component - Neomorphism Style (Softer Colors)
const TriviaCard = ({ onTriviaClick }) => {
  const prefersReducedMotion = useReducedMotion();

  const handleClick = () => {
    onTriviaClick();
  };

  return (
    <div
      className="p-3 mb-4 rounded-2xl cursor-pointer transition-all duration-300 ease-in-out hover:scale-105"
      style={{
        background: '#fafafa',
        boxShadow: '6px 6px 12px #e8e8e8, -6px -6px 12px #ffffff'
      }}
      onClick={handleClick}
    >
      <div className="flex flex-col items-center">
        <div
          className="p-2 rounded-full mb-2 transition-all duration-300 ease-in-out hover:scale-110"
          style={{
            background: '#fafafa',
            boxShadow: 'inset 3px 3px 6px #e8e8e8, inset -3px -3px 6px #ffffff'
          }}
        >
          <BookOpen className="w-4 h-4 sm:w-6 sm:h-6 lg:w-7 lg:h-7 text-blue-500" />
        </div>
        <p className="text-center text-xs sm:text-sm lg:text-base font-sarabun text-gray-600">คลิกเพื่อดูเกร็ดความรู้</p>
      </div>
    </div>
  );
};

// Export all components
export {
  Header,
  MonitoringPanel,
  MapSection,
  SimpleChevronBar,
  TriviaCard,
  Footer,
  HistoryData,
};