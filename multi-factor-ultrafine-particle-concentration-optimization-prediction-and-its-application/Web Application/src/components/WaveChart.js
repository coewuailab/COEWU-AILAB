// components/WaveChart.js
'use client';

import React, { useState, useEffect } from 'react';

const WaveChart = ({ data, pmType, getBarColor }) => {
    const [animationKey, setAnimationKey] = useState(0);

    // Trigger animation when data changes
    useEffect(() => {
        setAnimationKey(prev => prev + 1);
    }, [data, pmType]);

    if (!data || data.length === 0) return null;

    // Get the maximum value for scaling
    const maxValue = Math.max(...data.map(item => {
        const value = pmType === 'PM01' ? item.pc01 : pmType === 'PM2.5' ? item.pm25 : item.pm10;
        return value;
    }));

    const chartHeight = 300;
    const chartWidth = 100;

    return (
        <div className="relative bg-gradient-to-b from-blue-50 to-blue-100 rounded-lg overflow-hidden p-4" style={{ height: '400px' }}>
            {/* Y-axis label */}
            <div className="absolute top-4 right-4 text-sm text-gray-600">
                μg/m³
            </div>

            {/* Chart container */}
            <div className="absolute inset-0 flex items-end justify-around p-8 pt-16">
                {data.map((item, index) => {
                    const value = pmType === 'PM01' ? item.pc01 : pmType === 'PM2.5' ? item.pm25 : item.pm10;
                    const height = (value / (maxValue + 10)) * chartHeight;
                    const color = getBarColor(value, pmType);

                    return (
                        <div key={`${item.time}-${index}`} className="flex flex-col items-center">
                            <div className="relative">
                                {/* Wave bar */}
                                <div
                                    className="rounded-full transition-all duration-1000 ease-in-out wave-animation"
                                    style={{
                                        width: '24px',
                                        height: `${height}px`,
                                        backgroundColor: color,
                                        animation: `wave-${index} ${2 + index * 0.2}s ease-in-out infinite alternate`,
                                        boxShadow: `0 0 10px ${color}30`,
                                        transform: 'translateY(0)',
                                    }}
                                />

                                {/* Value label */}
                                <div
                                    className="absolute bg-white px-2 py-1 rounded text-xs font-bold shadow-lg z-10 transform -translate-x-1/2 transition-all duration-300"
                                    style={{
                                        top: '-30px',
                                        left: '50%',
                                        border: `2px solid ${color}`,
                                        color: color,
                                        animation: `float-${index} ${2 + index * 0.2}s ease-in-out infinite alternate`,
                                    }}
                                >
                                    {value.toFixed(1)}
                                </div>
                            </div>

                            {/* X-axis labels */}
                            <div className="text-center mt-4">
                                <span className="text-xs text-gray-700 font-medium block">{item.time}</span>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Dynamic CSS animations */}
            <style jsx>{`
        ${data.map((_, index) => `
          @keyframes wave-${index} {
            0% { 
              transform: scaleY(1) scaleX(1);
              opacity: 0.8;
            }
            50% { 
              transform: scaleY(1.08) scaleX(1.02);
              opacity: 0.9;
            }
            100% { 
              transform: scaleY(1.15) scaleX(1.05);
              opacity: 1;
            }
          }
          
          @keyframes float-${index} {
            0% { 
              transform: translateX(-50%) translateY(0px);
            }
            50% { 
              transform: translateX(-50%) translateY(-2px);
            }
            100% { 
              transform: translateX(-50%) translateY(-4px);
            }
          }
        `).join('')}
        
        .wave-animation {
          border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%;
        }
        
        .wave-animation:hover {
          transform: scaleY(1.2) scaleX(1.1) !important;
          transition: all 0.3s ease-in-out;
        }
      `}</style>
        </div>
    );
};

export default WaveChart;