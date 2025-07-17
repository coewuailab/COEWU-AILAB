'use client'

import React, { useState, useEffect, useRef } from 'react';
import { getDatabase, ref, get, onValue } from 'firebase/database';
import { initializeApp, getApps } from 'firebase/app';
import { LOCATION_CONFIGS } from '../config/firebase-configs';
import { Cloud } from 'lucide-react';
import {
    getAirQualityColor,
    determinePM25Status,
    determinePM10Status,
    determineHourlyMeanPC01Status,
    determineDailyMeanPC01Status,
    formatPCValue
} from '../data/monitoring-data';

const HistoryData = ({ selectedLocation }) => {
    const [dailyMeanPC01, setDailyMeanPC01] = useState('0.00');
    const [dailyMeanPM10, setDailyMeanPM10] = useState('0.00');
    const [dailyMeanPM25, setDailyMeanPM25] = useState('0.00');
    const [hourlyMeanPC01, setHourlyMeanPC01] = useState('0.00');
    const [hourlyMeanPM10, setHourlyMeanPM10] = useState('0.00');
    const [hourlyMeanPM25, setHourlyMeanPM25] = useState('0.00');
    const [dayType, setDayType] = useState('Weekend');
    const [days, setDays] = useState('Sunday');
    const [range, setRange] = useState('morning');
    const [loading, setLoading] = useState(true);
    const firebaseAppRef = useRef({});
    const dataListenerRef = useRef(null);

    const locationId = selectedLocation?.id || 'cafe-amazon-st';
    const locationConfig = LOCATION_CONFIGS[locationId];
    console.log('Active Location ID:', locationId);
    console.log('Active Location Config:', locationConfig);

    if (!locationConfig) {
        console.error('Location config not found for ID:', locationId);
        return <div>Configuration error: Location data not found</div>;
    }

    const getFirebaseApp = (locationConfig) => {
        if (!firebaseAppRef.current[locationConfig.id]) {
            try {
                const existingApp = getApps().find((app) => app.name === locationConfig.id);
                if (!existingApp) {
                    firebaseAppRef.current[locationConfig.id] = initializeApp(locationConfig.firebaseConfig, locationConfig.id);
                    console.log('Firebase app initialized for:', locationConfig.id);
                } else {
                    firebaseAppRef.current[locationConfig.id] = existingApp;
                    console.log('Firebase app reused for:', locationConfig.id);
                }
            } catch (error) {
                console.error('Error initializing Firebase app:', error);
                throw error;
            }
        }
        return firebaseAppRef.current[locationConfig.id];
    };

    const fetchData = (locationConfig) => {
        setLoading(true);
        resetStates(); // Reset states before fetching new data
        try {
            const firebaseApp = getFirebaseApp(locationConfig);
            const db = getDatabase(firebaseApp);
            const today = new Date().toISOString().split('T')[0]; // 2025-06-18
            const fullDataPath = `Testing/${locationConfig.testingPath}/${today}`;
            console.log('Fetching data path for', locationConfig.name, ':', fullDataPath);
            const dataRef = ref(db, fullDataPath);

            if (dataListenerRef.current) {
                dataListenerRef.current();
            }

            get(dataRef).then((snapshot) => {
                if (snapshot.exists()) {
                    const raw = snapshot.val();
                    console.log('Raw Firebase data for', locationConfig.name, ':', raw);
                    const timestamps = Object.keys(raw);
                    if (timestamps.length > 0) {
                        const latestTimestamp = timestamps.reduce((latest, current) => {
                            try {
                                const latestTime = new Date(`2025-06-18 ${latest.split(':').join(':')}`).getTime();
                                const currentTime = new Date(`2025-06-18 ${current.split(':').join(':')}`).getTime();
                                console.log('Comparing timestamps:', { latest, current, latestTime, currentTime });
                                return currentTime > latestTime ? current : latest;
                            } catch (e) {
                                console.error('Timestamp parsing error for', locationConfig.name, ':', e, 'Timestamp:', current);
                                return latest;
                            }
                        }, timestamps[0]);
                        const latestData = raw[latestTimestamp] || {};
                        console.log('Extracted latest data for', locationConfig.name, ':', latestData);
                        setDailyMeanPC01(latestData['Daily_mean_PC01'] !== undefined ? Number(latestData['Daily_mean_PC01']).toFixed(2) : '0.00');
                        setDailyMeanPM10(latestData['Daily_mean_PM10'] !== undefined ? Number(latestData['Daily_mean_PM10']).toFixed(2) : '0.00');
                        setDailyMeanPM25(latestData['Daily_mean_PM25'] !== undefined ? Number(latestData['Daily_mean_PM25']).toFixed(2) : '0.00');
                        setHourlyMeanPC01(latestData['Hourly_mean_PC01'] !== undefined ? Number(latestData['Hourly_mean_PC01']).toFixed(2) : '0.00');
                        setHourlyMeanPM10(latestData['Hourly_mean_PM10'] !== undefined ? Number(latestData['Hourly_mean_PM10']).toFixed(2) : '0.00');
                        setHourlyMeanPM25(latestData['Hourly_mean_PM25'] !== undefined ? Number(latestData['Hourly_mean_PM25']).toFixed(2) : '0.00');
                        setDayType(latestData['DayType'] ?? 'Weekend');
                        setDays(latestData['Days'] ?? 'Sunday');

                        const hour = new Date(`2025-06-18 ${latestTimestamp}`).getHours() || 19; // 07:17 PM
                        setRange(hour >= 7 && hour < 17 ?
                            (hour < 9 ? 'early_morning' :
                                hour < 11 ? 'morning' :
                                    hour < 13 ? 'early_afternoon' :
                                        hour < 15 ? 'afternoon' :
                                            'late_afternoon') : 'Other');
                    } else {
                        console.log('No timestamps found for', locationConfig.name);
                    }
                } else {
                    console.log('No data available at path for', locationConfig.name, ':', fullDataPath);
                }
                setLoading(false);
            }).catch((error) => {
                console.error('Error fetching initial data for', locationConfig.name, ':', error);
                setLoading(false);
            });

            dataListenerRef.current = onValue(dataRef, (snapshot) => {
                if (snapshot.exists()) {
                    const raw = snapshot.val();
                    console.log('Real-time data for', locationConfig.name, ':', raw);
                    const timestamps = Object.keys(raw);
                    if (timestamps.length > 0) {
                        const latestTimestamp = timestamps.reduce((latest, current) => {
                            try {
                                const latestTime = new Date(`2025-06-18 ${latest.split(':').join(':')}`).getTime();
                                const currentTime = new Date(`2025-06-18 ${current.split(':').join(':')}`).getTime();
                                return currentTime > latestTime ? current : latest;
                            } catch (e) {
                                console.error('Real-time timestamp parsing error for', locationConfig.name, ':', e, 'Timestamp:', current);
                                return latest;
                            }
                        }, timestamps[0]);
                        const latestData = raw[latestTimestamp] || {};
                        console.log('Real-time latest data for', locationConfig.name, ':', latestData);
                        setDailyMeanPC01(latestData['Daily_mean_PC01'] !== undefined ? Number(latestData['Daily_mean_PC01']).toFixed(2) : '0.00');
                        setDailyMeanPM10(latestData['Daily_mean_PM10'] !== undefined ? Number(latestData['Daily_mean_PM10']).toFixed(2) : '0.00');
                        setDailyMeanPM25(latestData['Daily_mean_PM25'] !== undefined ? Number(latestData['Daily_mean_PM25']).toFixed(2) : '0.00');
                        setHourlyMeanPC01(latestData['Hourly_mean_PC01'] !== undefined ? Number(latestData['Hourly_mean_PC01']).toFixed(2) : '0.00');
                        setHourlyMeanPM10(latestData['Hourly_mean_PM10'] !== undefined ? Number(latestData['Hourly_mean_PM10']).toFixed(2) : '0.00');
                        setHourlyMeanPM25(latestData['Hourly_mean_PM25'] !== undefined ? Number(latestData['Hourly_mean_PM25']).toFixed(2) : '0.00');
                        setDayType(latestData['DayType'] ?? 'Weekend');
                        setDays(latestData['Days'] ?? 'Sunday');

                        const hour = new Date(`2025-06-18 ${latestTimestamp}`).getHours() || 19; // 07:17 PM
                        setRange(hour >= 7 && hour < 17 ?
                            (hour < 9 ? 'early_morning' :
                                hour < 11 ? 'morning' :
                                    hour < 13 ? 'early_afternoon' :
                                        hour < 15 ? 'afternoon' :
                                            'late_afternoon') : 'Other');
                    }
                }
            }, (error) => {
                console.error('Error in real-time listener for', locationConfig.name, ':', error);
            });
        } catch (err) {
            console.error('Initialization error for', locationConfig.name, ':', err);
            setLoading(false);
        }
    };

    const resetStates = () => {
        setDailyMeanPC01('0.00');
        setDailyMeanPM10('0.00');
        setDailyMeanPM25('0.00');
        setHourlyMeanPC01('0.00');
        setHourlyMeanPM10('0.00');
        setHourlyMeanPM25('0.00');
        setDayType('Weekend');
        setDays('Sunday');
        setRange('morning');
    };

    useEffect(() => {
        if (locationConfig) {
            fetchData(locationConfig);
        }
    }, [locationId, selectedLocation, locationConfig]);

    const currentDate = new Date().toLocaleString('th-TH', {
        weekday: 'long',
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
        timeZone: 'Asia/Bangkok',
    }).replace(/(\d+)\/(\d+)\/(\d+)/, '$2 $1 $3'); // วันพุธ 18 มิ.ย. 2568 19:17

    const pm25StatusHourly = determinePM25Status(hourlyMeanPM25);
    const pm10StatusHourly = determinePM10Status(hourlyMeanPM10);
    const pc01StatusHourly = determineHourlyMeanPC01Status(hourlyMeanPC01);
    const pm25ColorHourly = getAirQualityColor(pm25StatusHourly);
    const pm10ColorHourly = getAirQualityColor(pm25StatusHourly);
    const pc01ColorHourly = getAirQualityColor(pc01StatusHourly);

    const pm25StatusDaily = determinePM25Status(dailyMeanPM25);
    const pm10StatusDaily = determinePM10Status(dailyMeanPM10);
    const pc01StatusDaily = determineDailyMeanPC01Status(dailyMeanPC01);
    const pm25ColorDaily = getAirQualityColor(pm25StatusDaily);
    const pm10ColorDaily = getAirQualityColor(pm25StatusDaily);
    const pc01ColorDaily = getAirQualityColor(pc01StatusDaily);

    return (
        // <div className="w-full bg-white">
        <div className="rounded-lg shadow-lg p-6 flex flex-col gap-6">
            {/* Header with title on left and subtitle on right */}
            <div className="flex justify-between items-center mb-3 sm:mb-4">
                <h2 className="text-base sm:text-lg lg:text-xl xl:text-2xl font-medium text-gray-800 font-sarabun">ข้อมูลย้อนหลัง</h2>
                <div className="text-sm text-gray-500 font-sarabun">
                    {loading ? 'กำลังโหลด...' : `สถานที่: ${locationConfig.name}`}
                </div>
            </div>
            {/* Hourly Card */}
            <div className="rounded-lg p-4 bg-gray-50">
                <div className="mb-4">
                    <div className="text-2xl font-semibold font-sarabun text-gray-800">Hourly</div>
                </div>
                <div className="flex flex-row gap-4 w-full">
                    {/* PC0.1 Sub-Card */}
                    <div className="flex-1 rounded-lg p-4 flex flex-col items-center border border-gray-200" style={{ backgroundColor: pc01ColorHourly }}>
                        <div className="flex items-center gap-3">
                            <Cloud className="w-6 h-6 text-white" />
                            <span className="text-xl font-bold text-white font-montserrat">PC0.1</span>
                        </div>
                        <span className="text-2xl font-bold text-white mt-2">
                            {loading ? 'Loading...' : Number(hourlyMeanPC01).toFixed(2)}
                        </span>
                        <span className="text-lg font-bold text-white mt-1">
                            {loading ? 'Loading...' : 'PNC'}
                        </span>
                        <p className="text-sm text-white opacity-90 mt-2">Ultrafine particles</p>
                    </div>
                    {/* PM2.5 Sub-Card */}
                    <div className="flex-1 rounded-lg p-4 flex flex-col items-center border border-gray-200" style={{ backgroundColor: pm25ColorHourly }}>
                        <div className="flex items-center gap-3">
                            <Cloud className="w-6 h-6 text-white" />
                            <span className="text-xl font-bold text-white font-montserrat">PM2.5</span>
                        </div>
                        <span className="text-2xl font-bold text-white mt-2">
                            {loading ? 'Loading...' : hourlyMeanPM25}
                        </span>
                        <span className="text-lg font-bold text-white mt-1">
                            {loading ? 'Loading...' : 'µg/m³'}
                        </span>
                        <p className="text-sm text-white opacity-90 mt-2">Fine particulate matter</p>
                    </div>
                    {/* PM10 Sub-Card */}
                    <div className="flex-1 rounded-lg p-4 flex flex-col items-center border border-gray-200" style={{ backgroundColor: pm10ColorHourly }}>
                        <div className="flex items-center gap-3">
                            <Cloud className="w-6 h-6 text-white" />
                            <span className="text-xl font-bold text-white font-montserrat">PM10</span>
                        </div>
                        <span className="text-2xl font-bold text-white mt-2">
                            {loading ? 'Loading...' : hourlyMeanPM10}
                        </span>
                        <span className="text-lg font-bold text-white mt-1">
                            {loading ? 'Loading...' : 'µg/m³'}
                        </span>
                        <p className="text-sm text-white opacity-90 mt-2">Coarse particulate matter</p>
                    </div>
                </div>
            </div>
            {/* Daily Card */}
            <div className="rounded-lg p-4 bg-gray-50">
                <div className="mb-4">
                    <div className="text-2xl font-semibold font-sarabun text-gray-800">Daily</div>
                </div>
                <div className="flex flex-row gap-4 w-full">
                    {/* PC0.1 Sub-Card */}
                    <div className="flex-1 rounded-lg p-4 flex flex-col items-center border border-gray-200" style={{ backgroundColor: pc01ColorDaily }}>
                        <div className="flex items-center gap-3">
                            <Cloud className="w-6 h-6 text-white" />
                            <span className="text-xl font-bold text-white font-montserrat">PC0.1</span>
                        </div>
                        <span className="text-2xl font-bold text-white mt-2">
                            {loading ? 'Loading...' : Number(dailyMeanPC01).toFixed(2)}
                        </span>
                        <span className="text-lg font-bold text-white mt-1">
                            {loading ? 'Loading...' : 'PNC'}
                        </span>
                        <p className="text-sm text-white opacity-90 mt-2">Ultrafine particles</p>
                    </div>
                    {/* PM2.5 Sub-Card */}
                    <div className="flex-1 rounded-lg p-4 flex flex-col items-center border border-gray-200" style={{ backgroundColor: pm25ColorDaily }}>
                        <div className="flex items-center gap-3">
                            <Cloud className="w-6 h-6 text-white" />
                            <span className="text-xl font-bold text-white font-montserrat">PM2.5</span>
                        </div>
                        <span className="text-2xl font-bold text-white mt-2">
                            {loading ? 'Loading...' : dailyMeanPM25}
                        </span>
                        <span className="text-lg font-bold text-white mt-1">
                            {loading ? 'Loading...' : 'µg/m³'}
                        </span>
                        <p className="text-sm text-white opacity-90 mt-2">Fine particulate matter</p>
                    </div>
                    {/* PM10 Sub-Card */}
                    <div className="flex-1 rounded-lg p-4 flex flex-col items-center border border-gray-200" style={{ backgroundColor: pm10ColorDaily }}>
                        <div className="flex items-center gap-3">
                            <Cloud className="w-6 h-6 text-white" />
                            <span className="text-xl font-bold text-white font-montserrat">PM10</span>
                        </div>
                        <span className="text-2xl font-bold text-white mt-2">
                            {loading ? 'Loading...' : dailyMeanPM10}
                        </span>
                        <span className="text-lg font-bold text-white mt-1">
                            {loading ? 'Loading...' : 'µg/m³'}
                        </span>
                        <p className="text-sm text-white opacity-90 mt-2">Coarse particulate matter</p>
                    </div>
                </div>
            </div>
        </div>
            // </div>
    );
};

export default HistoryData;