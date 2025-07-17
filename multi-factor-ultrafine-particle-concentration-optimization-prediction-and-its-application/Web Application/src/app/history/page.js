'use client';

import React, { useState, useEffect, useRef, useMemo, Component } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { getDatabase, ref, onValue } from 'firebase/database';
import { initializeApp, getApps } from 'firebase/app';
import { Header, Footer } from '../../components/MonitoringInterface';
import dynamic from 'next/dynamic';
import { LOCATION_CONFIGS } from '../../config/firebase-configs';
import { getAirQualityColor, PM_THRESHOLDS, determineHourlyMeanPC01Status, determinePM25Status, determinePM10Status } from '../../data/monitoring-data';
import { isEqual } from 'lodash';

const EnhancedMultiBarChart = dynamic(() => import('../../components/EnhancedMultiBarChart'), { ssr: false });

// Error Boundary Component
class ErrorBoundary extends Component {
    state = { hasError: false, error: null };

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="w-full h-full flex items-center justify-center bg-red-50">
                    <div className="text-center">
                        <div className="text-red-600 text-6xl mb-4">⚠️</div>
                        <p className="text-red-600 font-medium thai-text">เกิดข้อผิดพลาดในการแสดงผลหน้า</p>
                        <p className="text-red-500 text-sm mt-1 english-text">{this.state.error?.message || 'Unknown error'}</p>
                        <button
                            onClick={() => window.location.reload()}
                            className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
                        >
                            <span className="thai-text">ลองใหม่</span>
                        </button>
                    </div>
                </div>
            );
        }
        return this.props.children;
    }
}

// Air quality status functions
const getBarColor = (value, type) => {
    const status =
        type === 'PC0.1'
            ? determineHourlyMeanPC01Status(value)
            : type === 'PM2.5'
                ? determinePM25Status(value)
                : determinePM10Status(value);
    return getAirQualityColor(status);
};

const getSelectedStatus = (selectedData, pmType) => {
    if (!selectedData) return { status: 'Good', range: '', color: getAirQualityColor('Good') };
    const value = selectedData[pmType === 'PC0.1' ? 'pc01' : pmType === 'PM2.5' ? 'pm25' : 'pm10'];
    const status =
        pmType === 'PC0.1'
            ? determineHourlyMeanPC01Status(value)
            : pmType === 'PM2.5'
                ? determinePM25Status(value)
                : determinePM10Status(value);
    const thresholds =
        pmType === 'PC0.1' ? PM_THRESHOLDS.HourlyMeanPC01 : pmType === 'PM2.5' ? PM_THRESHOLDS.PM : PM_THRESHOLDS.PM10;
    return {
        status,
        range: getRangeText(status, thresholds, pmType),
        color: getAirQualityColor(status),
    };
};

const getRangeText = (status, thresholds, pmType) => {
    switch (status) {
        case 'Good':
            return pmType === 'PC0.1' ? `0-${thresholds.Good} PNC` : `0-${thresholds.Good} μg/m³`;
        case 'Warning':
            return pmType === 'PC0.1' ? `${thresholds.Good + 1}-${thresholds.Warning} PNC` : `${thresholds.Good + 0.1}-${thresholds.Warning} μg/m³`;
        case 'Affects health':
            return pmType === 'PC0.1'
                ? `${thresholds.Warning + 1}-${thresholds['Affects health']} PNC`
                : `${thresholds.Warning + 0.1}-${thresholds['Affects health']} μg/m³`;
        case 'Danger':
            return pmType === 'PC0.1'
                ? `${thresholds['Affects health'] + 1}-${thresholds.Danger} PNC`
                : `${thresholds['Affects health'] + 0.1}-${thresholds.Danger} μg/m³`;
        case 'Hazardous':
            return pmType === 'PC0.1' ? `${thresholds.Danger + 1}+ PNC` : `${thresholds.Danger + 0.1}+ μg/m³`;
        default:
            return '';
    }
};

const createFallbackData = (timestamp = '00:00') => ({
    time: timestamp,
    pc01: '0.00',
    pm25: '0.00',
    pm10: '0.00',
});

export default function HistoryPage() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const locationId = searchParams.get('locationId');
    const selectedLocation = useMemo(() => {
        console.log('useMemo: locationId =', locationId);
        const loc = locationId
            ? LOCATION_CONFIGS[locationId] || {
                name: 'Cafe Amazon ST',
                id: 'cafe-amazon-st',
                firebaseConfig: LOCATION_CONFIGS['cafe-amazon-st']?.firebaseConfig,
                testingPath: 'Cafe',
            }
            : {
                name: 'Cafe Amazon ST',
                id: 'cafe-amazon-st',
                firebaseConfig: LOCATION_CONFIGS['cafe-amazon-st']?.firebaseConfig,
                testingPath: 'Cafe',
            };
        console.log('useMemo: selectedLocation =', loc);
        return loc;
    }, [locationId]);

    const [timeFrame, setTimeFrame] = useState('Daily');
    const [pmType, setPmType] = useState('PM10');
    const [historicalData, setHistoricalData] = useState([createFallbackData()]);
    const [selectedData, setSelectedData] = useState(createFallbackData());
    const [loading, setLoading] = useState(true);
    const firebaseAppRef = useRef({});
    const prevHistoricalDataRef = useRef(historicalData);

    const handleBackToHome = async () => {
        try {
            console.log('Navigating to home...');
            await router.push('/');
        } catch (error) {
            console.error('Navigation error:', error);
        }
    };

    const handleLocationSelect = (locationId) => {
        console.log('handleLocationSelect: Selected location ID:', locationId);
        router.push(`/history?locationId=${locationId}`);
    };

    const getFirebaseApp = (locationConfig) => {
        console.log('getFirebaseApp: locationConfig =', locationConfig);
        if (!locationConfig?.id || !locationConfig?.firebaseConfig) {
            console.error('Invalid locationConfig:', locationConfig);
            return null;
        }
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
                return null;
            }
        }
        return firebaseAppRef.current[locationConfig.id];
    };

    useEffect(() => {
        console.log('useEffect: Running with selectedLocation =', selectedLocation, 'timeFrame =', timeFrame);
        if (!selectedLocation || !selectedLocation.id) {
            console.error('No location selected or invalid selectedLocation:', selectedLocation);
            setHistoricalData([createFallbackData()]);
            setSelectedData(createFallbackData());
            setLoading(false);
            return;
        }

        setLoading(true);
        const locationConfig = LOCATION_CONFIGS[selectedLocation.id];
        if (!locationConfig) {
            console.error('Location config not found for ID:', selectedLocation.id);
            setHistoricalData([createFallbackData()]);
            setSelectedData(createFallbackData());
            setLoading(false);
            return;
        }

        let unsubscribe;
        try {
            const firebaseApp = getFirebaseApp(locationConfig);
            if (!firebaseApp) {
                console.error('Firebase app initialization failed for:', locationConfig.id);
                setHistoricalData([createFallbackData()]);
                setSelectedData(createFallbackData());
                setLoading(false);
                return;
            }
            const db = getDatabase(firebaseApp);
            const today = new Date().toISOString().split('T')[0];
            const fullDataPath = `Testing/${locationConfig.testingPath}/${today}`;
            console.log('Fetching data path for', locationConfig.name, ':', fullDataPath);
            const dataRef = ref(db, fullDataPath);

            unsubscribe = onValue(
                dataRef,
                (snapshot) => {
                    console.log('onValue: Raw snapshot:', snapshot.val());
                    if (snapshot.exists()) {
                        const raw = snapshot.val();
                        const timestamps = Object.keys(raw);
                        if (timestamps.length > 0) {
                            const formattedData = timestamps
                                .map((timestamp) => {
                                    const data = raw[timestamp] || {};
                                    return {
                                        time: timestamp,
                                        pc01:
                                            data[timeFrame === 'Hourly' ? 'Hourly_mean_PC01' : 'Daily_mean_PC01'] !== undefined
                                                ? Number(data[timeFrame === 'Hourly' ? 'Hourly_mean_PC01' : 'Daily_mean_PC01']).toFixed(2)
                                                : '0.00',
                                        pm25:
                                            data[timeFrame === 'Hourly' ? 'Hourly_mean_PM25' : 'Daily_mean_PM25'] !== undefined
                                                ? Number(data[timeFrame === 'Hourly' ? 'Hourly_mean_PM25' : 'Daily_mean_PM25']).toFixed(2)
                                                : '0.00',
                                        pm10:
                                            data[timeFrame === 'Hourly' ? 'Hourly_mean_PM10' : 'Daily_mean_PM10'] !== undefined
                                                ? Number(data[timeFrame === 'Hourly' ? 'Hourly_mean_PM10' : 'Daily_mean_PM10']).toFixed(2)
                                                : '0.00',
                                    };
                                })
                                .sort((a, b) => {
                                    const aTime = new Date(`${today} ${a.time}`).getTime();
                                    const bTime = new Date(`${today} ${b.time}`).getTime();
                                    return aTime - bTime;
                                });

                            if (!isEqual(formattedData, prevHistoricalDataRef.current)) {
                                console.log('onValue: Updating historicalData with', formattedData);
                                setHistoricalData(formattedData);
                                setSelectedData(formattedData[formattedData.length - 1] || createFallbackData());
                                prevHistoricalDataRef.current = formattedData;
                            } else {
                                console.log('onValue: No data change, skipping state update');
                            }
                        } else {
                            console.log('onValue: No timestamps found for', locationConfig.name);
                            if (!isEqual([createFallbackData()], prevHistoricalDataRef.current)) {
                                setHistoricalData([createFallbackData()]);
                                setSelectedData(createFallbackData());
                                prevHistoricalDataRef.current = [createFallbackData()];
                            }
                        }
                        setLoading(false);
                    } else {
                        console.log('onValue: No data available at path for', locationConfig.name, ':', fullDataPath);
                        if (!isEqual([createFallbackData()], prevHistoricalDataRef.current)) {
                            setHistoricalData([createFallbackData()]);
                            setSelectedData(createFallbackData());
                            prevHistoricalDataRef.current = [createFallbackData()];
                        }
                        setLoading(false);
                    }
                },
                (error) => {
                    console.error('onValue: Firebase error:', error.code, error.message);
                    if (!isEqual([createFallbackData()], prevHistoricalDataRef.current)) {
                        setHistoricalData([createFallbackData()]);
                        setSelectedData(createFallbackData());
                        prevHistoricalDataRef.current = [createFallbackData()];
                    }
                    setLoading(false);
                }
            );
        } catch (err) {
            console.error('useEffect: Initialization error for', locationConfig?.name, ':', err);
            if (!isEqual([createFallbackData()], prevHistoricalDataRef.current)) {
                setHistoricalData([createFallbackData()]);
                setSelectedData(createFallbackData());
                prevHistoricalDataRef.current = [createFallbackData()];
            }
            setLoading(false);
        }

        return () => {
            console.log('useEffect: Cleaning up Firebase listener');
            if (unsubscribe) {
                unsubscribe();
            }
        };
    }, [selectedLocation, timeFrame]);

    const getFormattedDateTime = (timeFrame, selectedData) => {
        if (!selectedData || !selectedData.time) {
            const now = new Date();
            return timeFrame === 'Daily'
                ? now.toLocaleString('en-US', {
                    weekday: 'short',
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric',
                }).replace(/,/, '')
                : now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }) +
                ' ' +
                now.toLocaleString('en-US', {
                    weekday: 'short',
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric',
                }).replace(/,/, '');
        }

        const [hours, minutes] = selectedData.time.split(':');
        const dataDate = new Date();
        dataDate.setHours(hours, minutes, 0, 0);

        if (timeFrame === 'Hourly') {
            return `${selectedData.time} ${dataDate.toLocaleString('en-US', {
                weekday: 'short',
                month: 'short',
                day: 'numeric',
                year: 'numeric',
            }).replace(/,/, '')}`;
        } else if (timeFrame === 'Daily') {
            return dataDate.toLocaleString('en-US', {
                weekday: 'short',
                month: 'short',
                day: 'numeric',
                year: 'numeric',
            }).replace(/,/, '');
        }
        return selectedData.time;
    };

    const selectedStatus = getSelectedStatus(selectedData, pmType);

    return (
        <ErrorBoundary>
            <div className="flex flex-col min-h-screen w-full" style={{ backgroundColor: '#FFFFFF' }}>
                <Header selectedLocation={selectedLocation} onClick={handleBackToHome} />
                <main className="flex-1 w-full max-w-none p-4">
                    <div className="bg-white rounded-lg shadow-sm p-6 w-full" style={{ backgroundColor: '#F0F0F0' }}>
                        <div className="mb-6">
                            <h2 className="text-2xl font-bold english-text text-gray-800 mb-2">Air Quality History</h2>
                            <p className="text-gray-600 text-sm thai-text">
                                ข้อมูลย้อนหลังของ {selectedLocation?.name || 'Unknown Location'}
                            </p>
                        </div>
                        <div className="flex flex-row gap-8 items-start justify-between">
                            {/* Summary on the Left - ขยายขนาด */}
                            <div className="bg-white rounded-lg p-3 shadow-sm w-96 flex flex-row items-stretch border border-gray-200">
                                {/* Date/Time and Value on the left */}
                                <div className="flex-1 flex flex-col items-center justify-center p-2">
                                    <div className="text-sm english-text text-gray-600 mb-2">{getFormattedDateTime(timeFrame, selectedData)}</div>
                                    <div className="text-2xl font-bold english-text text-gray-900">
                                        {loading ? 'Loading...' : selectedData ? selectedData[pmType === 'PC0.1' ? 'pc01' : pmType === 'PM2.5' ? 'pm25' : 'pm10'] : '0.00'}
                                    </div>
                                    <div className="text-sm english-text text-gray-600">
                                        {pmType === 'PC0.1' ? 'PNC' : 'μg/m³'}
                                    </div>
                                </div>
                                {/* Criteria (Status) on the right, full block with rounded edge */}
                                <div className="w-1/3 flex items-center justify-center rounded-r-lg" style={{ backgroundColor: selectedStatus.color }}>
                                    <div className="text-white font-semibold english-text text-2xl text-center p-2">
                                        {loading ? 'Loading...' : selectedStatus.status}
                                    </div>
                                </div>
                            </div>

                            {/* Selection Bars on the Right */}
                            <div className="flex flex-col gap-3 w-auto min-w-fit">
                                <select
                                    value={selectedLocation.id}
                                    onChange={(e) => handleLocationSelect(e.target.value)}
                                    className="px-3 py-2 rounded-lg font-semibold text-white focus:outline-none text-sm w-40"
                                    style={{ backgroundColor: '#2DC653' }}
                                >
                                    <option value="cafe-amazon-st" className="english-text">Cafe Amazon ST</option>
                                    <option value="building-c4" className="thai-text">อาคารวิชาการ 4</option>
                                </select>
                                <select
                                    value={timeFrame}
                                    onChange={(e) => setTimeFrame(e.target.value)}
                                    className="px-3 py-2 rounded-lg font-semibold text-white focus:outline-none text-sm w-40"
                                    style={{ backgroundColor: '#2DC653' }}
                                >
                                    <option value="Daily" className="english-text">Daily</option>
                                    <option value="Hourly" className="english-text">Hourly</option>
                                </select>
                                <select
                                    value={pmType}
                                    onChange={(e) => setPmType(e.target.value)}
                                    className="px-3 py-2 rounded-lg font-semibold text-white focus:outline-none text-sm w-40"
                                    style={{ backgroundColor: '#2DC653' }}
                                >
                                    <option value="PC0.1" className="english-text">PC0.1</option>
                                    <option value="PM2.5" className="english-text">PM2.5</option>
                                    <option value="PM10" className="english-text">PM10</option>
                                </select>
                            </div>
                        </div>
                        <div className="bg-white rounded-lg shadow-sm w-full mt-4" style={{ height: '60vh' }}>
                            {loading ? (
                                <div className="text-center p-4 thai-text">Loading data...</div>
                            ) : (
                                <EnhancedMultiBarChart
                                    data={historicalData}
                                    pmType={pmType}
                                    getBarColor={getBarColor}
                                    onBarSelect={setSelectedData}
                                    timeFrame={timeFrame}
                                />
                            )}
                        </div>
                    </div>
                </main>
                <Footer />
            </div>
        </ErrorBoundary>
    );
}
