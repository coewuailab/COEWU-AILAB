'use client'

import { initializeApp } from 'firebase/app';
import { getDatabase, ref, onValue, query, orderByKey, limitToLast } from 'firebase/database';
import { useState, useEffect } from 'react';
import { LOCATION_CONFIGS } from '../config/firebase-configs';

// ===================================================================
// Firebase Configuration และ Constants
// ===================================================================

// ใช้ Firebase configuration จาก config file
const firebaseConfig = LOCATION_CONFIGS['cafe-amazon-st'].firebaseConfig || LOCATION_CONFIGS['building-c4']?.firebaseConfig;

// Initialize Firebase - เพิ่มการตรวจสอบ config
if (!firebaseConfig) {
  console.error('Firebase configuration not found in LOCATION_CONFIGS');
}

const app = initializeApp(firebaseConfig);
const database = getDatabase(app);

// Constants สำหรับ Firebase paths
const FIREBASE_USER_ID = 'gdRueJtWeNaMleXbEf4rWfuD6Kr1';
const PIERA_PATH = 'Piera/';
const RAW_PATH = 'RAWdata/';
const TESTING_PATH = 'Testing/';

// ===================================================================
// Helper Functions สำหรับการจัดการเวลาและการแสดงผล
// ===================================================================

const getCurrentDatePath = () => {
  const date = new Date();
  const year = date.getFullYear();
  const month = (date.getMonth() + 1).toString().padStart(2, '0');
  const day = date.getDate().toString().padStart(2, '0');
  return `${year}-${month}-${day}`;
};

const getCurrentTime = () => {
  const date = new Date();
  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');
  const seconds = date.getSeconds().toString().padStart(2, '0');
  return `${hours}:${minutes}:${seconds}`;
};

const formatDisplayDate = (date) => {
  const targetDate = date || new Date();
  const month = (targetDate.getMonth() + 1).toString().padStart(2, '0');
  const day = targetDate.getDate().toString().padStart(2, '0');
  const year = targetDate.getFullYear();
  return `${month}/${day}/${year}`;
};

// ===================================================================
// Utility Functions สำหรับการจัดการข้อมูลอย่างปลอดภัย
// ===================================================================

// ฟังก์ชันสำหรับแปลงค่าให้เป็นตัวเลขอย่างปลอดภัย
const safeParse = (value, defaultValue = 0) => {
  if (value === null || value === undefined || value === '') {
    return defaultValue;
  }

  const parsed = parseFloat(value);
  return isNaN(parsed) ? defaultValue : parsed;
};

// ฟังก์ชันสำหรับตรวจสอบว่าข้อมูลมีค่าที่มีประโยชน์หรือไม่
const hasValidData = (data) => {
  if (!data || typeof data !== 'object') {
    return false;
  }

  const pc01 = safeParse(data.PC01 || data.pc01);
  const pm01 = safeParse(data.PM01 || data.pm01);
  const pm25 = safeParse(data.PM25 || data.pm25 || data.PM);
  const pm10 = safeParse(data.PM10 || data.pm10 || data.PM100);
  const hourlyMeanPC01 = safeParse(data.hourlymeanpc0_1);
  const dailyMeanPC01 = safeParse(data.dailymeanpc0_1);

  return pc01 > 0 || pm01 > 0 || pm25 > 0 || pm10 > 0 || hourlyMeanPC01 > 0 || dailyMeanPC01 > 0;
};

// ===================================================================
// PM Details - ข้อมูลรายละเอียดสำหรับ PM Readings
// ===================================================================

export const PM_DETAILS = {
  'PM2.5': {
    title: 'PM2.5 (ฝุ่นละอองขนาดเล็ก≤ 2.5 ไมโครเมตร)',
    description: 'ฝุ่น PM2.5 คือฝุ่นละอองที่ลอยอยู่ในอากาศโดยวัดเส้นผ่านศูนย์กลางได้ 2.5 ไมโครเมตรหรือน้อยกว่านั้น PM2.5 มีขนาดเล็กมากกระทั่งมันสามารถถูกดูดซึมเข้าไปในกระแสเลือดได้เมื่อสูดหายใจเข้าไป ด้วยเหตุนี้ มันจึงเป็นสารมลพิษที่เป็นภัยต่อสุขภาพมากที่สุด',
    sources: 'แหล่งที่มาของมันอาจถูกส่งออกมาจากแหล่งที่มนุษย์สร้างขึ้นหรือแหล่งที่มาตามธรรมชาติก็ได้ หรืออาจถูกสร้างขึ้นโดยสารมลพิษอื่น การเผาไหม้ที่เป็นผลมาจากโรงงานพลังงาน ควันและเขม่าจากไฟป่าและการเผาขยะ การปล่อยมลพิษจากรถยนต์และการเผาไหม้จากมอเตอร์ กระบวนการทางอุตสาหกรรมที่เกี่ยวข้องกับปฏิกิริยาทางเคมีระหว่างก๊าซ(ซัลเฟอร์ไดออกไซด์ ไนโตรเจนออกไซด์ และสารประกอบอินทรีย์ระเหย)',
    'short-term': [
      'การระคายเคืองต่อดวงตา คอ และจมูก',
      'การเต้นของหัวใจที่ผิดปกติ',
      'โรคหอบหืด',
      'การไอ อาการแน่นหน้าอก และอาการหายใจลำบาก'
    ],
    'long-term': [
      'การอุดตันของเส้นโลหิตที่ไปเลี้ยงสมอง',
      'การเสียชีวิตก่อนวัยอันควร',
      'โรคระบบทางเดินหายใจ เช่น โรคหลอดลมอักเสบ โรคหอบหืด โรคถุงลมโป่งพอง',
      'ความเสียหายต่อเนื้อเยื่อปอด',
      'มะเร็ง',
      'โรคหัวใจ'
    ],
  },
  'PM10': {
    title: 'PM10 (อนุภาคหยาบ ≤ 10 ไมครอน)',
    description: 'PM10 คือฝุ่นละอองแขวนลอยในอากาศที่มีเส้นผ่านศูนย์กลาง 10 ไมโครเมตรหรือน้อยกว่า (รวมถึงควัน เขม่าควัน เกลือ กรด และโลหะ) ความแตกการอยู่ในขนาดของมัน่ PM10 นั้นหยาบและใหญ่กว่า PM2.5',
    sources: 'ฝุ่นผงจากการก่อสร้าง การถมที่ และเกษตรกรรม ฝุ่นผงที่ปลิวจากที่เปิด ควันจากไฟป่าและการเผาขยะ ปฏิกิริยาทางเคมีจากอุตสาหกรรม รถยนต์',
    'short-term': [
      'อาการหายใจลำบาก',
      'อาการเจ็บหน้าอก',
      'อาการอึดอัดในระบบทางเดินหายใจทั่วไป',
      'อาการเจ็บคอ',
      'อาการคัดจมูก'
    ],
    'long-term': [
      'ความเสียหายของเนื้อเยื่อปอด',
      'อาการหอบหืด',
      'การเสียชีวิตก่อนวัยอันควร'
    ]
  },
  'HourlyMeanPC0.1': {
    title: 'Hourly Mean PC0.1 (ค่าเฉลี่ยรายชั่วโมงของอนุภาคขนาดเล็ก)',
    description: 'ค่าเฉลี่ยรายชั่วโมงของจำนวนอนุภาคขนาดเล็ก (PC0.1) วัดเป็นอนุภาคต่อลูกบาศก์เซนติเมตร (PNC) หากระดับต่ำกว่า 20,000 อนุภาค/ลูกบาศก์เซนติเมตร จะใช้เกณฑ์เดียวกับ PC01 (Good, Warning, Affects health, Danger, Hazardous) หากระดับ ≥ 20,000 จะถือว่าเป็น Hazardous',
    sources: 'การเผาไหม้จากยานพาหนะ โรงงานอุตสาหกรรม ควันจากไฟป่า และกิจกรรมที่ก่อให้เกิดฝุ่น',
    'short-term': [
      'การระคายเคืองต่อระบบทางเดินหายใจ',
      'อาการไอหรือหายใจลำบาก'
    ],
    'long-term': [
      'ความเสี่ยงต่อโรคระบบทางเดินหายใจ'
    ]
  },
  'DailyMeanPC0.1': {
    title: 'Daily Mean PC0.1 (ค่าเฉลี่ยรายวันของอนุภาคขนาดเล็ก)',
    description: 'ค่าเฉลี่ยรายวันของจำนวนอนุภาคขนาดเล็ก (PC0.1) วัดเป็นอนุภาคต่อลูกบาศก์เซนติเมตร (PNC) ระดับสูงอาจบ่งชี้ถึงมลพิษในอากาศที่ต่อเนื่องและส่งผลต่อสุขภาพในระยะยาว',
    sources: 'การเผาไหม้จากยานพาหนะ โรงงานอุตสาหกรรม ควันจากไฟป่า และกิจกรรมที่ก่อให้เกิดฝุ่น',
    'short-term': [
      'การระคายเคืองต่อระบบทางเดินหายใจ'
    ],
    'long-term': [
      'ความเสี่ยงต่อโรคระบบทางเดินหายใจ',
      'ผลกระทบต่อสุขภาพหัวใจและหลอดเลือด'
    ]
  }
};

// ===================================================================
// Export Functions ที่ต้องใช้ใน component หลัก
// ===================================================================

export const getAirQualityColor = (status) => {
  const colors = {
    'Good': '#2DC653',
    'Warning': '#FECF3E',
    'Affects health': '#FF9500',
    'Danger': '#D02224',
    'Hazardous': '#973AA8',
  };
  return colors[status] || colors['Good'];
};

export const formatPMValue = (value) => {
  const numericValue = parseFloat(value);
  if (isNaN(numericValue)) return value;
  return `${numericValue.toFixed(2)} μg/m³`;
};

export const formatPCValue = (value) => {
  const numericValue = parseFloat(value);
  if (isNaN(numericValue)) return value;
  return `${numericValue.toFixed(0)} PNC`;
};

// PM thresholds สำหรับการประเมินคุณภาพอากาศ
export const PM_THRESHOLDS = {
  PC01: {
    Good: 260,
    Warning: 540,
    'Affects health': 3620,
    Danger: 6270,
    Hazardous: Number.MAX_VALUE
  },
  PM01: {
    Good: 0.6,
    Warning: 0.9,
    'Affects health': 1.1,
    Danger: 1.6,
    Hazardous: Number.MAX_VALUE
  },
  PM: {
    Good: 15.0,
    Warning: 37.5,
    'Affects health': 75.0,
    Danger: 150.0,
    Hazardous: Number.MAX_VALUE
  },
  PM10: { //สำหรับแสดงในแถบ simplechevronbar เท่านั้นไม่ได้นำมาคิดเพื่อ่คำนวณสถานะ//
    Good: 50,
    Warning: 80,
    'Affects health': 120,
    Danger: 180,
    Hazardous: Number.MAX_VALUE
  },

  HourlyMeanPC01: {
    Good: 258,
    Warning: 543,
    'Affects health': 3616,
    Danger: 6271,
    Hazardous: 20000
  },
  DailyMeanPC01: {
    Good: 1000,
    'Affects health': 3616,
    Danger: 6271,
    Hazardous: 10000
  },
};

// ===================================================================
// Individual PM Status Functions - ฟังก์ชันคำนวณสถานะแต่ละค่า PM
// ===================================================================

export const determinePC01Status = (pc01Value) => {
  const pc01Val = safeParse(pc01Value);

  if (pc01Val > PM_THRESHOLDS.PC01.Danger) {
    return "Hazardous";
  } else if (pc01Val > PM_THRESHOLDS.PC01['Affects health']) {
    return "Danger";
  } else if (pc01Val > PM_THRESHOLDS.PC01.Warning) {
    return "Affects health";
  } else if (pc01Val > PM_THRESHOLDS.PC01.Good) {
    return "Warning";
  } else {
    return "Good";
  }
};

export const determinePM01Status = (pm01Value) => {
  const pm01Val = safeParse(pm01Value);

  if (pm01Val > PM_THRESHOLDS.PM01.Danger) {
    return "Hazardous";
  } else if (pm01Val > PM_THRESHOLDS.PM01['Affects health']) {
    return "Danger";
  } else if (pm01Val > PM_THRESHOLDS.PM01.Warning) {
    return "Affects health";
  } else if (pm01Val > PM_THRESHOLDS.PM01.Good) {
    return "Warning";
  } else {
    return "Good";
  }
};

export const determinePM25Status = (pm25Value) => {
  const pm25Val = safeParse(pm25Value);

  console.log('=== PM2.5 STATUS CALCULATION ===');
  console.log('PM2.5 Value:', pm25Val);
  console.log('Thresholds:', PM_THRESHOLDS.PM);

  if (pm25Val > PM_THRESHOLDS.PM.Danger) {
    console.log('PM2.5 Status: Hazardous');
    return "Hazardous";
  } else if (pm25Val > PM_THRESHOLDS.PM['Affects health']) {
    console.log('PM2.5 Status: Danger');
    return "Danger";
  } else if (pm25Val > PM_THRESHOLDS.PM.Warning) {
    console.log('PM2.5 Status: Affects health');
    return "Affects health";
  } else if (pm25Val > PM_THRESHOLDS.PM.Good) {
    console.log('PM2.5 Status: Warning');
    return "Warning";
  } else {
    console.log('PM2.5 Status: Good');
    return "Good";
  }
};

export const determinePM10Status = (pm10Value) => {
  const pm10Val = safeParse(pm10Value);

  console.log('=== PM10 STATUS CALCULATION ===');
  console.log('PM10 Value:', pm10Val);
  console.log('Thresholds:', PM_THRESHOLDS.PM10);

  if (pm10Val > PM_THRESHOLDS.PM10.Danger) {
    console.log('PM10 Status: Hazardous');
    return "Hazardous";
  } else if (pm10Val > PM_THRESHOLDS.PM10['Affects health']) {
    console.log('PM10 Status: Danger');
    return "Danger";
  } else if (pm10Val > PM_THRESHOLDS.PM10.Warning) {
    console.log('PM10 Status: Affects health');
    return "Affects health";
  } else if (pm10Val > PM_THRESHOLDS.PM10.Good) {
    console.log('PM10 Status: Warning');
    return "Warning";
  } else {
    console.log('PM10 Status: Good');
    return "Good";
  }
};

export const determineHourlyMeanPC01Status = (hourlyMeanValue) => {
  const hourlyVal = safeParse(hourlyMeanValue);

  console.log('=== HOURLY MEAN PC0.1 STATUS CALCULATION ===');
  console.log('Hourly Mean PC0.1 Value:', hourlyVal);
  console.log('Thresholds:', PM_THRESHOLDS.HourlyMeanPC01);

  if (hourlyVal >= PM_THRESHOLDS.HourlyMeanPC01.Hazardous) {
    console.log('Hourly Mean PC0.1 Status: Hazardous');
    return "Hazardous";
  } else {
    const status = determinePC01Status(hourlyVal);
    console.log('Hourly Mean PC0.1 Status (using PC01 thresholds):', status);
    return status;
  }
};

export const determineDailyMeanPC01Status = (dailyMeanValue) => {
  const dailyVal = safeParse(dailyMeanValue);

  console.log('=== DAILY MEAN PC0.1 STATUS CALCULATION ===');
  console.log('Daily Mean PC0.1 Value:', dailyVal);
  console.log('Thresholds:', PM_THRESHOLDS.DailyMeanPC01);

  if (dailyVal > PM_THRESHOLDS.DailyMeanPC01.Hazardous) {
    console.log('Daily Mean PC0.1 Status: Hazardous');
    return "Hazardous";
  } else {
    console.log('Daily Mean PC0.1 Status: Good');
    return "Good";
  }
};

export const getPMReadingStatusAndColor = (pmReading) => {
  if (!pmReading || !pmReading.type || pmReading.value === undefined) {
    return {
      status: 'Good',
      color: getAirQualityColor('Good'),
      thresholds: PM_THRESHOLDS.PM
    };
  }

  let status = 'Good';
  let thresholds = {};

  switch (pmReading.type) {
    case 'PC01':
      status = determinePC01Status(pmReading.value);
      thresholds = PM_THRESHOLDS.PC01;
      break;
    case 'PM0.1':
      status = determinePM01Status(pmReading.value);
      thresholds = PM_THRESHOLDS.PM01;
      break;
    case 'PM2.5':
      status = determinePM25Status(pmReading.value);
      thresholds = PM_THRESHOLDS.PM25;
      break;
    case 'PM10':
      status = determinePM10Status(pmReading.value);
      thresholds = PM_THRESHOLDS.PM10;
      break;
    case 'HourlyMeanPC0.1':
      status = determineHourlyMeanPC01Status(pmReading.value);
      thresholds = PM_THRESHOLDS.HourlyMeanPC01;
      break;
    case 'DailyMeanPC0.1':
      status = determineDailyMeanPC01Status(pmReading.value);
      thresholds = PM_THRESHOLDS.DailyMeanPC01;
      break;
    default:
      status = 'Good';
      thresholds = PM_THRESHOLDS.PM;
  }

  const color = getAirQualityColor(status);

  console.log(`=== ${pmReading.type} STATUS ===`);
  console.log('Value:', pmReading.value);
  console.log('Status:', status);
  console.log('Color:', color);
  console.log('Thresholds:', thresholds);
  console.log('==========================');

  return { status, color, thresholds };
};

export const getPMDetails = (pmType) => {
  return PM_DETAILS[pmType] || {
    title: 'ไม่พบข้อมูล',
    description: 'ไม่มีข้อมูลสำหรับประเภท PM นี้',
    healthImpact: 'ไม่ระบุ',
    recommendation: ['ไม่ระบุ'],
    sources: 'ไม่ระบุ',
    measurementUnit: 'ไม่ระบุ'
  };
};

// ===================================================================
// Overall Air Quality Function - ฟังก์ชันคำนวณสถานะรวม
// ===================================================================

export const determineAirQuality = (pc01, pm01, pm25, pm10, hourlyMeanPC01, dailyMeanPC01) => {
  try {
    const pc01Val = safeParse(pc01);
    const pm01Val = safeParse(pm01);
    const pm25Val = safeParse(pm25);
    const pm10Val = safeParse(pm25);
    const hourlyMeanPC01Val = safeParse(hourlyMeanPC01);
    const dailyMeanPC01Val = safeParse(dailyMeanPC01);

    console.log('=== DETERMINE AIR QUALITY DEBUG ===');
    console.log('Input values:', { pc01, pm01, pm25, pm10, hourlyMeanPC01, dailyMeanPC01 });
    console.log('Parsed values:', { pc01Val, pm01Val, pm25Val, pm10Val, hourlyMeanPC01Val, dailyMeanPC01Val });

    if (
      pc01Val > PM_THRESHOLDS.PC01.Danger ||
      hourlyMeanPC01Val >= PM_THRESHOLDS.HourlyMeanPC01.Hazardous ||
      dailyMeanPC01Val > PM_THRESHOLDS.DailyMeanPC01.Hazardous
    ) {
      console.log('Status: Hazardous');
      console.log('===================================');
      return "Hazardous";
    } else if (
      pc01Val > PM_THRESHOLDS.PC01['Affects health'] ||
      hourlyMeanPC01Val > PM_THRESHOLDS.HourlyMeanPC01['Affects health']
    ) {
      console.log('Status: Danger');
      console.log('===================================');
      return "Danger";
    } else if (
      pc01Val > PM_THRESHOLDS.PC01.Warning ||
      hourlyMeanPC01Val > PM_THRESHOLDS.HourlyMeanPC01.Warning
    ) {
      console.log('Status: Affects health');
      console.log('===================================');
      return "Affects health";
    } else if (
      pc01Val > PM_THRESHOLDS.PC01.Good ||
      hourlyMeanPC01Val > PM_THRESHOLDS.HourlyMeanPC01.Good
    ) {
      console.log('Status: Warning');
      console.log('===================================');
      return "Warning";
    } else {
      console.log('Status: Good');
      console.log('===================================');
      return "Good";
    }
  } catch (error) {
    console.error('Error in determineAirQuality:', error);
    return "Good";
  }
};

// ===================================================================
// ฟังก์ชันสำหรับการสร้างข้อมูลที่ปลอดภัยและครบถ้วน
// ===================================================================

const transformToComponentFormat = (data, locationName, dataSource = 'testing') => {
  if (!data || typeof data !== 'object') {
    console.warn('Invalid data received in transformToComponentFormat:', data);
    return createFallbackData(locationName, dataSource);
  }

  try {
    const pc01 = safeParse(data.pc01 || data.PC01 || data['PC01']);
    const pm01 = safeParse(data.pm01 || data.PM01);
    const pm25 = safeParse(data.pm25 || data.PM25 || data.PM);
    const pm10 = safeParse(data.pm10 || data.PM10 || data.PM100);
    const hourlyMeanPC01 = safeParse(data.hourlymeanpc0_1);
    const dailyMeanPC01 = safeParse(data.dailymeanpc0_1);

    const temperature = safeParse(data.temperature || data.IndoorTemperature, 25.5);
    const humidity = safeParse(data.humidity || data.IndoorHumidity, 65);

    console.log('=== TRANSFORM DEBUG ===');
    console.log('Raw Firebase Data:', data);
    console.log('Parsed Values:');
    console.log('  PC01:', pc01);
    console.log('  PM01:', pm01);
    console.log('  PM25:', pm25);
    console.log('  PM10:', pm10);
    console.log('  HourlyMeanPC0.1:', hourlyMeanPC01);
    console.log('  DailyMeanPC0.1:', dailyMeanPC01);

    const overallStatus = determineAirQuality(pc01, pm01, pm25, pm10, hourlyMeanPC01, dailyMeanPC01);
    console.log('Overall Status:', overallStatus);

    const pc01Status = determinePC01Status(pc01);
    const pm01Status = determinePM01Status(pm01);
    const pm25Status = determinePM25Status(pm25);
    const pm10Status = determinePM25Status(pm25);
    const hourlyMeanPC01Status = determineHourlyMeanPC01Status(hourlyMeanPC01);
    const dailyMeanPC01Status = determineDailyMeanPC01Status(dailyMeanPC01);

    console.log('Individual Statuses:');
    console.log('  PC01 Status:', pc01Status);
    console.log('  PM01 Status:', pm01Status);
    console.log('  PM25 Status:', pm25Status);
    console.log('  PM10 Status:', pm10Status);
    console.log('  HourlyMeanPC0.1 Status:', hourlyMeanPC01Status);
    console.log('  DailyMeanPC0.1 Status:', dailyMeanPC01Status);

    const recommendations = getGeneralRecommendations(overallStatus);
    console.log('Recommendations:', recommendations);
    console.log('=======================');

    const transformedData = {
      date: formatDisplayDate(new Date()),
      time: data.timestamp || getCurrentTime(),
      location: locationName || 'Unknown Location',
      mainReading: {
        type: dataSource === 'testing' ? 'PC0.1' : 'PM0.1',
        value: dataSource === 'testing' ? pc01 : pm01,
        unit: dataSource === 'testing' ? 'PNC' : 'μg/m³',
        status: overallStatus
      },
      conditions: {
        temperature: `${temperature}°C`,
        humidity: `${humidity}%`
      },
      pmReadings: [
        {
          type: 'PC0.1',
          value: pc01,
          unit: 'PNC',
          status: pc01Status
        },
        {
          type: 'PM0.1',
          value: pm01,
          unit: 'μg/m³',
          status: pm01Status
        },
        {
          type: 'PM2.5',
          value: pm25,
          unit: 'μg/m³',
          status: pm25Status
        },
        {
          type: 'PM10',
          value: pm10,
          unit: 'μg/m³',
          status: pm10Status
        },
        {
          type: 'HourlyMeanPC0.1',
          value: hourlyMeanPC01,
          unit: 'PNC',
          status: hourlyMeanPC01Status
        },
        {
          type: 'DailyMeanPC0.1',
          value: dailyMeanPC01,
          unit: 'PNC',
          status: dailyMeanPC01Status
        }
      ],
      recommendations: recommendations,
      _rawData: data,
      _dataSource: dataSource,
      _isValid: hasValidData(data),
      _calculatedStatus: overallStatus,
      _individualStatuses: {
        pc01: pc01Status,
        pm01: pm01Status,
        pm25: pm25Status,
        pm10: pm10Status,
        hourlyMeanPC01: hourlyMeanPC01Status,
        dailyMeanPC01: dailyMeanPC01Status
      },
      _timestamp: new Date().toISOString()
    };

    console.log('=== TRANSFORMED DATA ===');
    console.log('Location:', locationName);
    console.log('Data Source:', dataSource);
    console.log('Main Reading:', transformedData.mainReading);
    console.log('PM Readings with Individual Status:', transformedData.pmReadings);
    console.log('Individual Statuses:', transformedData._individualStatuses);
    console.log('Is Valid Data:', hasValidData(data));
    console.log('========================');

    return transformedData;
  } catch (error) {
    console.error('Error in transformToComponentFormat:', error);
    return createFallbackData(locationName, dataSource);
  }
};

// ฟังก์ชันสำหรับสร้างข้อมูล fallback เมื่อไม่มีข้อมูลจริง
const createFallbackData = (locationName, dataSource = 'testing') => {
  const fallbackData = {
    date: formatDisplayDate(new Date()),
    time: getCurrentTime(),
    location: locationName || 'กำลังโหลด...',
    mainReading: {
      type: dataSource === 'testing' ? 'PC01' : 'PM0.1',
      value: 0,
      unit: dataSource === 'testing' ? 'PNC' : 'μg/m³',
      status: 'Good'
    },
    conditions: {
      temperature: '25.5°C',
      humidity: '65%'
    },
    pmReadings: [
      { type: 'PC01', value: 0, unit: 'PNC', status: 'Good' },
      { type: 'PM0.1', value: 0, unit: 'μg/m³', status: 'Good' },
      { type: 'PM2.5', value: 0, unit: 'μg/m³', status: 'Good' },
      { type: 'PM10', value: 0, unit: 'μg/m³', status: 'Good' },
      { type: 'HourlyMeanPC0.1', value: 0, unit: 'PNC', status: 'Good' },
      { type: 'DailyMeanPC0.1', value: 0, unit: 'PNC', status: 'Good' }
    ],
    recommendations: [
      'กำลังโหลดข้อมูล...',
      'กำลังเชื่อมต่อกับ Firebase...',
      'โปรดรอสักครู่...'
    ],
    _rawData: null,
    _dataSource: dataSource,
    _isValid: false,
    _isFallback: true
  };

  console.log('Created fallback data for:', locationName);
  return fallbackData;
};

// ===================================================================
// Hook Functions
// ===================================================================

export const useMonitoringData = () => {
  return useLocationMonitoringData({
    id: 'cafe-amazon-st',
    name: 'Cafe Amazon สาขา ST',
    dataSource: 'testing',
    pieraUserId: 'gdRueJtWeNaMleXbEf4rWfuD6Kr1',
    pieraPath: '',
    rawDataPath: 'Cafe',
    testingPath: 'Cafe'
  });
};

export const useLocationMonitoringData = (locationData) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!locationData || (!locationData.rawDataPath && !locationData.testingPath)) {
      console.error('Missing locationData or paths:', locationData);
      setError('ข้อมูลตำแหน่งไม่ครบถ้วน');
      setData(createFallbackData(locationData?.name));
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    const datePath = getCurrentDatePath();
    const dataSource = locationData.dataSource || 'testing';

    console.log('=== FIREBASE QUERY SETUP ===');
    console.log('Location:', locationData.name);
    console.log('Data Source:', dataSource);
    console.log('Date Path:', datePath);

    if (dataSource === 'testing') {
      const testingFirebasePath = `/Testing/${locationData.testingPath}/${datePath}`;
      const rawFirebasePath = `/RAWdata/${locationData.rawDataPath}/${datePath}`;

      console.log('Testing path:', testingFirebasePath);
      console.log('RAW path:', rawFirebasePath);

      let dataCollector = {
        testing: null,
        raw: null,
        testingReceived: false,
        rawReceived: false
      };

      const updateCombinedData = () => {
        if (!dataCollector.testingReceived && !dataCollector.rawReceived) {
          return;
        }

        try {
          const combinedData = {
            ...dataCollector.testing,
            ...dataCollector.raw
          };

          console.log('=== COMBINED DATA ===');
          console.log('Testing data:', dataCollector.testing);
          console.log('RAW data:', dataCollector.raw);
          console.log('Combined data:', combinedData);
          console.log('=====================');

          if (hasValidData(combinedData) || dataCollector.testingReceived) {
            const transformedData = transformToComponentFormat(
              combinedData,
              locationData.name,
              dataSource
            );
            setData(transformedData);
            setError(null);
          } else {
            const fallbackData = createFallbackData(locationData.name, dataSource);
            setData(fallbackData);
            setError('ไม่พบข้อมูลที่ใช้งานได้');
          }

          setLoading(false);
        } catch (error) {
          console.error('Error in updateCombinedData:', error);
          setData(createFallbackData(locationData.name, dataSource));
          setError('เกิดข้อผิดพลาดในการประมวลผลข้อมูล');
          setLoading(false);
        }
      };

      const testingQuery = query(
        ref(database, testingFirebasePath),
        orderByKey(),
        limitToLast(1)
      );

      const testingUnsubscribe = onValue(testingQuery, (snapshot) => {
        try {
          const timeData = snapshot.val();
          console.log('Testing data received:', timeData);

          dataCollector.testingReceived = true;

          if (timeData) {
            const latestTime = Object.keys(timeData)[0];
            dataCollector.testing = {
              ...timeData[latestTime],
              timestamp: latestTime
            };
          } else {
            console.log('No Testing data found');
            dataCollector.testing = {};
          }

          updateCombinedData();
        } catch (error) {
          console.error('Error processing testing data:', error);
          dataCollector.testingReceived = true;
          dataCollector.testing = {};
          updateCombinedData();
        }
      }, (err) => {
        console.error('Testing data error:', err);
        dataCollector.testingReceived = true;
        dataCollector.testing = {};
        updateCombinedData();
      });

      const rawQuery = query(
        ref(database, rawFirebasePath),
        orderByKey(),
        limitToLast(1)
      );

      const rawUnsubscribe = onValue(rawQuery, (snapshot) => {
        try {
          const timeData = snapshot.val();
          console.log('RAW data received:', timeData);

          dataCollector.rawReceived = true;

          if (timeData) {
            const allKeys = Object.keys(timeData);
            const sortedKeys = allKeys.sort();
            const latestTime = sortedKeys[sortedKeys.length - 1];
            dataCollector.raw = {
              ...timeData[latestTime],
              timestamp: latestTime
            };
          } else {
            console.log('No RAW data found');
            dataCollector.raw = {};
          }

          updateCombinedData();
        } catch (error) {
          console.error('Error processing raw data:', error);
          dataCollector.rawReceived = true;
          dataCollector.raw = {};
          updateCombinedData();
        }
      }, (err) => {
        console.error('RAW data error:', err);
        dataCollector.rawReceived = true;
        dataCollector.raw = {};
        updateCombinedData();
      });

      return () => {
        testingUnsubscribe();
        rawUnsubscribe();
      };

    } else if (dataSource === 'piera') {
      const pieraUserId = locationData.pieraUserId;
      const pieraSubPath = locationData.pieraPath || '';
      const pieraDataPath = pieraSubPath ?
        `${PIERA_PATH}${pieraUserId}/${pieraSubPath}/${datePath}` :
        `${PIERA_PATH}${pieraUserId}/${datePath}`;

      console.log('Piera path:', pieraDataPath);

      const pieraQuery = query(
        ref(database, pieraDataPath),
        orderByKey(),
        limitToLast(1)
      );

      const pieraUnsubscribe = onValue(pieraQuery, (snapshot) => {
        try {
          const timeData = snapshot.val();
          console.log('Piera data received:', timeData);

          if (timeData) {
            const latestTime = Object.keys(timeData)[0];
            const pieraData = {
              ...timeData[latestTime],
              timestamp: latestTime
            };

            const transformedData = transformToComponentFormat(
              pieraData,
              locationData.name,
              dataSource
            );

            setData(transformedData);
            setError(null);
          } else {
            console.log('No Piera data found');
            setData(createFallbackData(locationData.name, dataSource));
            setError('ไม่พบข้อมูล Piera');
          }
          setLoading(false);
        } catch (error) {
          console.error('Error processing piera data:', error);
          setData(createFallbackData(locationData.name, dataSource));
          setError('เกิดข้อผิดพลาดในการประมวลผลข้อมูล Piera');
          setLoading(false);
        }
      }, (err) => {
        console.error('Piera data error:', err);
        setError('เกิดข้อผิดพลาดในการดึงข้อมูล Piera');
        setData(createFallbackData(locationData.name, dataSource));
        setLoading(false);
      });

      return () => {
        pieraUnsubscribe();
      };

    } else if (dataSource === 'raw') {
      const rawFirebasePath = `${RAW_PATH}${locationData.rawDataPath}/${datePath}`;

      console.log('RAW path for all data:', rawFirebasePath);

      const rawQuery = query(
        ref(database, rawFirebasePath),
        orderByKey(),
        limitToLast(1)
      );

      const rawUnsubscribe = onValue(rawQuery, (snapshot) => {
        try {
          const timeData = snapshot.val();
          console.log('RAW data received:', timeData);

          if (timeData) {
            const latestTime = Object.keys(timeData)[0];
            const rawData = {
              ...timeData[latestTime],
              timestamp: latestTime
            };

            const transformedData = transformToComponentFormat(
              rawData,
              locationData.name,
              dataSource
            );

            setData(transformedData);
            setError(null);
          } else {
            console.log('No RAW data found');
            setData(createFallbackData(locationData.name, dataSource));
            setError('ไม่พบข้อมูล RAW');
          }
          setLoading(false);
        } catch (error) {
          console.error('Error processing raw data:', error);
          setData(createFallbackData(locationData.name, dataSource));
          setError('เกิดข้อผิดพลาดในการประมวลผลข้อมูล RAW');
          setLoading(false);
        }
      }, (err) => {
        console.error('RAW data error:', err);
        setError('เกิดข้อผิดพลาดในการดึงข้อมูล RAW');
        setData(createFallbackData(locationData.name, dataSource));
        setLoading(false);
      });

      return () => {
        rawUnsubscribe();
      };
    }

    console.log('========================');
  }, [locationData?.id, locationData?.dataSource]);

  return { data, loading, error };
};

// ===================================================================
// Recommendation Functions
// ===================================================================

export const getGeneralRecommendations = (airQualityStatus) => {
  switch (airQualityStatus) {
    case 'Good':
      return [
        "ไม่มีผลต่อสุขภาพสามารถใช้ชีวิตได้ตามปกติ",
        "สามารถเปิดหน้าต่างระบายอากาศได้"
      ];
    case 'Warning':
      return [
        "ไม่มีผลต่อสุขภาพสามารถใช้ชีวิตได้ตามปกติ",
        "สามารถเปิดหน้าต่างระบายอากาศได้"
      ];
    case 'Affects health':
      return [
        "มีความเสี่ยงต่อการระคายเคืองต่อดวงตา ผิวหนัง และลำคอ รวมถึงปัญหาระบบทางเดินหายใจ",
        "หลีกเลี่ยงการทำกิจกรรมที่ก่อให้เกิดฝุ่น",
        "สังเกตอาการผิดปกติของตนเอง"
      ];
    case 'Danger':
      return [
        "มีความเสี่ยงต่อการระคายเคืองต่อดวงตา ผิวหนัง และลำคอ รวมถึงปัญหาระบบทางเดินหายใจ",
        "หลีกเลี่ยงพื้นที่ หรือกิจกรรมที่ทำให้เกิดฝุ่น",
        "สังเกตอาการของตนเอง",
        "หากมีอาการควรไปพบแพทย์",
        "ติดตั้งเครื่องฟอกอากาศ"
      ];
    case 'Hazardous':
      return [
        "มีความเสี่ยงสูงที่จะเกิดการระคายเคืองอย่างรุนแรงและผลกระทบด้านลบต่อสุขภาพที่อาจกระตุ้นให้เกิดโรคหลอดเลือดหัวใจและระบบทางเดินหายใจ",
        "งดไปยังพื้นที่ หรือทำกิจกรรมที่ก่อให้เกิดฝุ่น",
        "สังเกตอาการของตนเอง",
        "หากมีอาการควรไปพบแพทย์",
        "ติดตั้งเครื่องฟอกอากาศ"
      ];
    default:
      return [
        "กำลังโหลดข้อมูล...",
        "กำลังวิเคราะห์คุณภาพอากาศ...",
        "กำลังประมวลผลคำแนะนำ..."
      ];
  }
};

export const getSensitiveRecommendations = (airQualityStatus) => {
  switch (airQualityStatus) {
    case 'Good':
      return [
        "ไม่มีผลต่อสุขภาพสามารถใช้ชีวิตได้ตามปกติ",
        "สามารถเปิดหน้าต่างระบายอากาศได้"
      ];
    case 'Warning':
      return [
        "สามารถเปิดหน้าต่างระบายอากาศได้",
        "มีความเสี่ยงต่อการระคายเคืองต่อดวงตา ผิวหนัง และลำคอ รวมถึงปัญหาระระบบทางเดินหายใจ",
        "สังเกตอาการของตนเอง",
        "หากมีอาการควรไปพบแพทย์",
      ];
    case 'Affects health':
      return [
        "มีความเสี่ยงต่อการระคายเคืองต่อดวงตา ผิวหนัง และลำคอ รวมถึงปัญหาระบบทางเดินหายใจ",
        "ปิดหน้าต่าง และหลีกเลี่ยงการถ่ายเทอากาศจากภายนอก",
        "สังเกตอาการของตนเอง",
        "หากมีอาการควรไปพบแพทย์",
      ];
    case 'Danger':
      return [
        "มีความเสี่ยงต่อการระคายเคืองอย่างรุนแรงต่อดวงตา ผิวหนัง และลำคอ รวมถึงปัญหาระบบทางเดินหายใจ",
        "หลีกเลี่ยงพื้นที่ หรือการทำกิจกรรมที่ก่อให้เกิดฝุ่น",
        "สังเกตอาการของตนเอง",
        "หากมีอาการควรไปพบแพทย์",
        "เตรียมยาหรืออุปกรณ์ตามคำสั่งแพทย์",
        "ติดตั้งเครื่องฟอกอากาศ"
      ];
    case 'Hazardous':
      return [
        "มีความเสี่ยงสูงที่จะเกิดการระคายเคืองอย่างรุนแรงและผลกระทบด้านลบต่อสุขภาพที่อาจกระตุ้นให้เกิดโรคหลอดเลือดหัวใจและระบบทางเดินหายใจ",
        "งดไปยังพื้นที่ หรือทำกิจกรรมที่ก่อให้เกิดฝุ่น",
        "สังเกตอาการของตนเอง",
        "หากมีอาการควรไปพบแพทย์",
        "เตรียมยาหรืออุปกรณ์ตามคำสั่งแพทย์",
        "ติดตั้งเครื่องฟอกอากาศ"
      ];
    default:
      return [
        "กำลังโหลดข้อมูล...",
        "กำลังวิเคราะห์คุณภาพอากาศ...",
        "กำลังประมวลผลคำแนะนำ..."
      ];
  }
};

export const getRecommendationIcon = (recommendation) => {
  const text = recommendation.toLowerCase();

  if (text.includes('ไม่มีผลต่อสุขภาพ') && text.includes('ใช้ชีวิตได้ตามปกติ')) {
    return '/assets/images/work.png';
  }
  if (text.includes('หน้าต่าง')) {
    return '/assets/images/window.png';
  }
  if (text.includes('หัวใจ')) {
    return '/assets/images/heart.png';
  }
  if (text.includes('ผิวหนัง') || text.includes('ดวงตา')) {
    return '/assets/images/irritation.png';
  }
  if (text.includes('สังเกต') || text.includes('ความเสี่ยง')) {
    return '/assets/images/Observe.png';
  }
  if (text.includes('งด') || text.includes('หลีกเลี่ยง')) {
    return '/assets/images/Nodust.png';
  }
  if (text.includes('เตรียมยาหรืออุปกรณ์ตามคำสั่งแพทย์')) {
    return '/assets/images/medicine.png';
  }
  if (text.includes('ติดตั้งเครื่องฟอกอากาศ')) {
    return '/assets/images/air-purifier.png';
  }
  if (text.includes('หากมีอาการให้ไปพบแพทย์') || text.includes('ไปพบแพทย์')) {
    return '/assets/images/Doctor.png';
  }
  if (text.includes('กำลังโหลด') || text.includes('กำลังวิเคราะห์') || text.includes('กำลังประมวลผล')) {
    return '⏳';
  }

  return '•';
};