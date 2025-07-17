'use client';

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { isEqual } from 'lodash';
import {
    PM_THRESHOLDS,
    getAirQualityColor,
    determineHourlyMeanPC01Status,
    determinePM25Status,
    determinePM10Status
} from '../data/monitoring-data';

const EnhancedMultiBarChart = ({ data: initialData, pmType, getBarColor, onBarSelect, timeFrame }) => {
    const [hoveredBar, setHoveredBar] = useState(null);
    const [chartDimensions, setChartDimensions] = useState({ width: 0, height: 0 });
    const [isInitialLoad, setIsInitialLoad] = useState(true);
    const chartRef = useRef(null);
    const barsRef = useRef(null);

    const CHART_PADDING = {
        top: 20,
        right: 100,
        bottom: 60,
        left: 20
    };

    const calculateYAxisIntervals = useMemo(() => {
        if (!initialData?.length) return [0, 100];

        const maxValue = Math.max(...initialData.map(item =>
            parseFloat(item[pmType === 'PC0.1' ? 'pc01' : pmType === 'PM2.5' ? 'pm25' : 'pm10'] || 0)
        ));
        const minValue = Math.min(...initialData.map(item =>
            parseFloat(item[pmType === 'PC0.1' ? 'pc01' : pmType === 'PM2.5' ? 'pm25' : 'pm10'] || 0)
        ));

        const thresholds = pmType === 'PC0.1' ? [
            0,
            PM_THRESHOLDS.HourlyMeanPC01.Good,
            PM_THRESHOLDS.HourlyMeanPC01.Warning,
            PM_THRESHOLDS.HourlyMeanPC01['Affects health'],
            PM_THRESHOLDS.HourlyMeanPC01.Danger,
            PM_THRESHOLDS.HourlyMeanPC01.Hazardous
        ] : pmType === 'PM2.5' ? [
            0,
            PM_THRESHOLDS.PM.Good,
            PM_THRESHOLDS.PM.Warning,
            PM_THRESHOLDS.PM['Affects health'],
            PM_THRESHOLDS.PM.Danger,
            PM_THRESHOLDS.PM.Hazardous
        ] : [
            0,
            PM_THRESHOLDS.PM10.Good,
            PM_THRESHOLDS.PM10.Warning,
            PM_THRESHOLDS.PM10['Affects health'],
            PM_THRESHOLDS.PM10.Danger,
            PM_THRESHOLDS.PM10.Hazardous
        ];

        let relevantIntervals = thresholds.filter(val => val >= minValue && val <= maxValue);
        if (relevantIntervals.length === 0 || !relevantIntervals.includes(0)) {
            relevantIntervals.unshift(0);
        }
        if (maxValue > thresholds[thresholds.length - 1]) {
            relevantIntervals.push(thresholds.findLast(val => val <= maxValue) || maxValue);
        }

        const roundedMax = pmType === 'PC0.1'
            ? Math.ceil(maxValue / 1000) * 1000
            : Math.ceil(maxValue / 10) * 10;

        return [...new Set([...relevantIntervals, roundedMax])].sort((a, b) => a - b);
    }, [initialData, pmType]);

    const displayedData = useMemo(() => {
        if (!initialData?.length) return [];
        const maxBars = 28;
        console.log('Initial Data length:', initialData.length, 'timeFrame:', timeFrame, 'maxValue:', Math.max(...initialData.map(item => parseFloat(item[pmType === 'PC0.1' ? 'pc01' : pmType === 'PM2.5' ? 'pm25' : 'pm10'] || 0))));
        return initialData.length > maxBars ? initialData.slice(-maxBars) : initialData;
    }, [initialData, timeFrame, pmType]);

    useEffect(() => {
        const handleResize = () => {
            if (chartRef.current) {
                const parent = chartRef.current.parentElement;
                const width = parent.clientWidth - (CHART_PADDING.left + CHART_PADDING.right);
                const height = parent.clientHeight - (CHART_PADDING.top + CHART_PADDING.bottom);

                setChartDimensions({
                    width: width,
                    height: Math.max(height, 400)
                });
            }
        };

        handleResize();
        window.addEventListener('resize', handleResize);
        const timer = setTimeout(() => setIsInitialLoad(false), 1000);
        return () => {
            window.removeEventListener('resize', handleResize);
            clearTimeout(timer);
        };
    }, [displayedData.length]);

    const chartHeight = chartDimensions.height || 400;
    const effectiveChartHeight = chartHeight - CHART_PADDING.top - CHART_PADDING.bottom;
    const minBarWidth = 20;
    const maxBarWidth = 50;
    const minBarSpacing = 10;

    const barWidth = Math.min(
        maxBarWidth,
        Math.max(minBarWidth, Math.floor((chartDimensions.width - CHART_PADDING.right) / Math.min(displayedData.length, 28)) - minBarSpacing)
    );
    const barSpacing = Math.max(minBarSpacing, Math.floor(barWidth * 0.2));
    const totalChartWidth = Math.min(displayedData.length, 28) * (barWidth + barSpacing) - barSpacing;

    const containerWidth = chartDimensions.width || Math.max(
        chartRef.current?.parentElement.clientWidth || 800,
        totalChartWidth + CHART_PADDING.right + CHART_PADDING.left + 100
    );

    const offsetX = containerWidth - totalChartWidth - CHART_PADDING.right;

    const getBarHeight = useCallback((value) => {
        const maxValue = Math.max(...calculateYAxisIntervals);
        return value === 0 ? 0 : Math.max(1, (value / maxValue) * effectiveChartHeight);
    }, [effectiveChartHeight, calculateYAxisIntervals]);

    const getYPosition = useCallback((value) => {
        const maxValue = Math.max(...calculateYAxisIntervals);
        return CHART_PADDING.top + ((maxValue - value) / maxValue) * effectiveChartHeight;
    }, [effectiveChartHeight, calculateYAxisIntervals]);

    const getStatus = useCallback((value) => {
        const val = parseFloat(value);
        return pmType === 'PC0.1'
            ? determineHourlyMeanPC01Status(val)
            : pmType === 'PM2.5'
                ? determinePM25Status(val)
                : determinePM10Status(val);
    }, [pmType]);

    const handleBarHover = useCallback((dataPoint, value, event) => {
        const rect = chartRef.current?.getBoundingClientRect();
        if (rect) {
            const yPos = event.clientY - rect.top;
            setHoveredBar({ dataPoint, value, yPos });
        }
    }, []);

    const handleBarLeave = useCallback(() => {
        setHoveredBar(null);
    }, []);

    const handleBarClick = useCallback((dataPoint) => {
        if (onBarSelect) {
            onBarSelect(dataPoint);
        }
    }, [onBarSelect]);

    const formatXLabel = useCallback((time) => {
        if (!time) return 'No Time';
        console.log('Raw time value:', time, 'timeFrame:', timeFrame);
        const date = new Date(`2025-07-16 ${time}`);
        if (isNaN(date.getTime())) return `Invalid: ${time}`;
        if (timeFrame === 'Hourly') {
            const hours = date.getHours().toString().padStart(2, '0');
            const minutes = date.getMinutes().toString().padStart(2, '0');
            return `${hours}:${minutes}`;
        }
        if (timeFrame === 'Daily') {
            const month = date.toLocaleString('en-US', { month: 'short' });
            const day = date.getDate().toString().padStart(2, '0');
            return `${month}/${day}`;
        }
        return 'No Time';
    }, [timeFrame]);

    const getUnitText = useCallback(() => {
        switch (pmType) {
            case 'PC0.1':
                return 'PNC';
            case 'PM2.5':
            case 'PM10':
                return 'μg/m³';
            default:
                return '';
        }
    }, [pmType]);

    const renderYAxis = useCallback(() => {
        const intervals = calculateYAxisIntervals;
        const yAxisBuffer = 20;
        const yAxisOffset = offsetX + totalChartWidth + yAxisBuffer;

        return (
            <div className="absolute h-full" style={{ width: `${CHART_PADDING.right}px`, left: `${yAxisOffset}px` }}>
                <div
                    className="absolute h-full border-l border-gray-300"
                    style={{ left: '0', top: '0' }}
                />

                {intervals.map((value) => (
                    <div
                        key={value}
                        className="absolute w-full"
                        style={{
                            top: `${getYPosition(value)}px`,
                        }}
                    >
                        <div className="w-full flex items-center justify-end">
                            <span className="text-xs font-medium text-gray-600" style={{
                                position: 'absolute',
                                right: '2px',
                                transform: 'translateY(-50%)',
                                paddingLeft: '2px'
                            }}>
                                {value.toLocaleString()}
                            </span>

                            <div
                                className="absolute border-t"
                                style={{
                                    width: `${totalChartWidth + yAxisBuffer}px`,
                                    right: `${containerWidth - yAxisOffset}px`,
                                    borderColor: value === 0 ? '#333' : '#e5e5e5',
                                    borderStyle: value === 0 ? 'solid' : 'dashed'
                                }}
                            />
                        </div>
                    </div>
                ))}

                <div
                    className="absolute text-sm text-gray-600 english-text"
                    style={{
                        left: '-15px',
                        top: '-22px',
                        transform: 'translateY(0)',
                        whiteSpace: 'nowrap'
                    }}
                >
                    {getUnitText()}
                </div>
            </div>
        );
    }, [calculateYAxisIntervals, getYPosition, totalChartWidth, getUnitText, offsetX, containerWidth]);

    if (!displayedData.length) {
        return (
            <div className="bg-white p-6 w-full h-full flex items-center justify-center">
                <span className="text-gray-500 english-text">No data available</span>
            </div>
        );
    }

    return (
        <div className="bg-white w-full h-full" style={{
            overflow: 'hidden',
            position: 'relative',
            minHeight: '400px',
            display: 'flex',
            flexDirection: 'column'
        }}>
            <div className="flex justify-between items-center mb-4 px-6 pt-6">
                <div className="flex items-center space-x-4">
                    <h3 className="text-lg font-semibold thai-text text-gray-800">กราฟคุณภาพอากาศย้อนหลัง</h3>
                    {hoveredBar && (
                        <div className="flex items-center space-x-2 animate-fade-in ml-4">
                            <div
                                className="w-4 h-4 rounded-full"
                                style={{ backgroundColor: getBarColor(hoveredBar.value, pmType) }}
                            />
                            <span className="text-sm font-medium english-text text-gray-700">
                                {hoveredBar.dataPoint.time}: {pmType} = {hoveredBar.value.toFixed(1)} {getUnitText()}
                            </span>
                            <span
                                className="text-xs px-2 py-1 rounded text-white font-medium"
                                style={{ backgroundColor: getBarColor(hoveredBar.value, pmType) }}
                            >
                                {getStatus(hoveredBar.value)}
                            </span>
                        </div>
                    )}
                </div>
            </div>

            <div className="flex-1 relative" style={{
                overflowX: 'auto',
                overflowY: 'hidden',
                padding: `${CHART_PADDING.top}px ${CHART_PADDING.right}px ${CHART_PADDING.bottom}px ${CHART_PADDING.left}px`,
                display: 'flex',
                justifyContent: 'center'
            }}>
                {renderYAxis()}

                <div
                    ref={chartRef}
                    className="relative flex items-end"
                    style={{
                        height: `${chartHeight - CHART_PADDING.bottom}px`,
                        width: `${containerWidth}px`,
                        minWidth: '100%'
                    }}
                >
                    <div ref={barsRef} className="flex items-end absolute bottom-0" style={{ left: `${offsetX}px` }}>
                        {displayedData.slice(0, 28).map((item, index) => {
                            const value = parseFloat(item[pmType === 'PC0.1' ? 'pc01' : pmType === 'PM2.5' ? 'pm25' : 'pm10'] || 0);
                            return (
                                <div
                                    key={`${item.time}-${pmType}-${index}`}
                                    className={`relative cursor-pointer transition-all duration-300 hover:scale-105 ${isInitialLoad ? 'rise-animation' : ''}`}
                                    style={{
                                        width: `${barWidth}px`,
                                        height: `${getBarHeight(value)}px`,
                                        backgroundColor: getBarColor(value, pmType),
                                        borderRadius: '4px 4px 0 0',
                                        marginRight: `${barSpacing}px`,
                                        boxShadow: hoveredBar?.dataPoint === item ? `0 0 10px ${getBarColor(value, pmType)}40` : 'none',
                                    }}
                                    onMouseEnter={(e) => handleBarHover(item, value, e)}
                                    onMouseLeave={handleBarLeave}
                                    onClick={() => handleBarClick(item)}
                                />
                            );
                        })}
                    </div>
                </div>

                <div
                    className="absolute bottom-0 flex"
                    style={{
                        left: `${offsetX}px`,
                        width: `${totalChartWidth}px`,
                        height: '40px'
                    }}
                >
                    {displayedData.slice(0, 28).map((item, index) => (
                        <div
                            key={`label-${index}`}
                            className="text-center"
                            style={{
                                width: `${barWidth}px`,
                                marginRight: `${barSpacing}px`,
                                transform: 'rotate(-45deg)',
                                transformOrigin: 'left top',
                                position: 'relative',
                                top: '10px'
                            }}
                        >
                            <div className="text-xs font-medium english-text text-gray-800 whitespace-nowrap">
                                {formatXLabel(item.time)}
                            </div>
                        </div>
                    ))}
                </div>

                <div
                    className="absolute bottom-0 w-full border-t border-gray-300"
                    style={{
                        left: `${offsetX}px`,
                        width: `${totalChartWidth}px`
                    }}
                />
            </div>

            <style jsx>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(-5px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fade-in {
          animation: fade-in 0.3s ease-out;
        }

        @keyframes rise {
          from {
            height: 0;
          }
          to {
            height: var(--bar-height);
          }
        }

        .rise-animation {
          animation: rise 1s ease-out forwards;
          --bar-height: ${getBarHeight(displayedData[0]?.[pmType === 'PC0.1' ? 'pc01' : pmType === 'PM2.5' ? 'pm25' : 'pm10'] || 0)}px;
        }

        .thai-text {
          font-family: var(--font-sarabun);
        }

        .english-text {
          font-family: var(--font-montserrat);
        }
      `}</style>
        </div>
    );
};

export default EnhancedMultiBarChart;