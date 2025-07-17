// src/config/firebase-configs.js

// Firebase configuration สำหรับ Cafe Amazon (Default)
export const CAFE_FIREBASE_CONFIG = {
    apiKey: "AIzaSyBqd5rgKGkO7h_Dn76ctkITkgkrvnV5tbE",
    authDomain: "ultrafine-particles.firebaseapp.com",
    databaseURL: "https://ultrafine-particles-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "ultrafine-particles",
    storageBucket: "ultrafine-particles.firebasestorage.app",
    messagingSenderId: "231538011955",
    appId: "1:231538011955:web:274e36a3c702324dc31846",
    measurementId: "G-KBQLGTKY0V"
};

// Firebase configuration สำหรับ อาคารวิชาการ 4 (ใช้เดียวกับ Cafe)
export const C4_FIREBASE_CONFIG = {
    apiKey: "AIzaSyBqd5rgKGkO7h_Dn76ctkITkgkrvnV5tbE",
    authDomain: "ultrafine-particles.firebaseapp.com",
    databaseURL: "https://ultrafine-particles-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "ultrafine-particles",
    storageBucket: "ultrafine-particles.firebasestorage.app",
    messagingSenderId: "231538011955",
    appId: "1:231538011955:web:274e36a3c702324dc31846",
    measurementId: "G-KBQLGTKY0V"
};

// Location data mapping
export const LOCATION_CONFIGS = {
    'cafe-amazon-st': {
        id: 'cafe-amazon-st', 
        name: 'Cafe Amazon สาขา ST',
        coords: [8.64437496101933, 99.89929488155569],
        firebaseConfig: CAFE_FIREBASE_CONFIG,
        dataSource: 'testing', // ประเภทข้อมูลหลัก
        pieraUserId: 'gdRueJtWeNaMleXbEf4rWfuD6Kr1',
        pieraPath: '', // ไม่มี sub-path (ใช้ root)
        testingPath: 'Cafe',      // <--- เพิ่ม path สำหรับ Testing (PC0.1)
        rawDataPath: 'Cafe',      // <--- เพิ่ม path สำหรับ RAWdata (PM2.5, PM10)
    },
    'building-c4': {
        id: 'building-c4',
        name: 'อาคารวิชาการ 4',
        coords: [8.638222, 99.897976],
        firebaseConfig: C4_FIREBASE_CONFIG,
        // C4 ใช้ RAWdata สำหรับทั้ง PM และ อุณหภูมิ/ความชื้น
        dataSource: 'testing', // ประเภทข้อมูล
        pieraUserId: null, // ไม่ใช้ Piera
        pieraPath: null,
        testingPath: 'Lab', // ใช้ Testing/Lab สำหรับทุกอย่าง
        rawDataPath: 'Lab' // ใช้ RAWdata/Lab สำหรับทุกอย่าง
    }
};