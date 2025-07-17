'use client'
import React, { useState, useEffect, useRef } from 'react';

const TRIVIA_DATA = {
    'PM2.5': {
        title: 'PM2.5 (ฝุ่นละอองขนาดเล็ก ≤ 2.5 ไมโครเมตร)',
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
        description: 'PM10 คือฝุ่นละอองแขวนลอยในอากาศที่มีเส้นผ่านศูนย์กลาง 10 ไมโครเมตรหรือน้อยกว่า (รวมถึงควัน เขม่าควัน เกลือ กรด และโลหะ) ความแตกต่างอยู่ในขนาดของมัน PM10 นั้นหยาบและใหญ่กว่า PM2.5',
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
        description: 'ค่าเฉลี่ยรายชั่วโมงของจำนวนอนุภาคขนาดเล็ก (PC0.1) วัดเป็นอนุภาคต่อลูกบาศก์เซนติเมตร (PNC) หากระดับต่ำกว่า 20,000 อนุภาค/ลูกบาศก์เซนติเมตร จะใช้เกณฑ์เดียวกับ PC0.1 (Good, Warning, Affects health, Danger, Hazardous) หากระดับ ≥ 20,000 จะถือว่าเป็น Hazardous',
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
    },
    'PCvsPM': [
        {
            title: 'PC และ PM คืออะไร?',
            description: [
                'PM (Particulate Matter) คือฝุ่นละอองที่มีขนาดต่าง ๆ วัดจากเส้นผ่านศูนย์กลางเป็นไมโครเมตร เช่น PM2.5, PM10',
                'PC (Particle Count) คือการนับจำนวนอนุภาคในอากาศต่อหน่วยปริมาตร (เช่น PC0.1) วัดเป็นอนุภาคต่อลูกบาศก์เซนติเมตร',
                'PM วัดมวลของฝุ่น ส่วน PC วัดจำนวนอนุภาค'
            ]
        },
        {
            title: 'ความแตกต่างระหว่าง PC และ PM',
            description: [
                'ขนาด: PM วัดขนาดอนุภาค (เช่น ≤2.5 หรือ ≤10 ไมโครเมตร) ส่วน PC วัดอนุภาคขนาดเล็กมาก (เช่น 0.1 ไมโครเมตร)',
                'หน่วยวัด: PM ใช้หน่วยไมโครกรัมต่อลูกบาศก์เมตร (μg/m³) ส่วน PC ใช้อนุภาคต่อลูกบาศก์เซนติเมตร (PNC)',
                'ผลกระทบ: PC0.1 อาจอันตรายกว่าเนื่องจากขนาดเล็กมาก สามารถแทรกซึมเข้าสู่ร่างกายได้ลึก'
            ]
        },
        {
            title: 'การใช้งานข้อมูล PC และ PM',
            description: [
                'PM: ใช้ประเมินคุณภาพอากาศทั่วไปและผลกระทบต่อสุขภาพ เช่น PM2.5 ที่เกี่ยวข้องกับโรคทางเดินหายใจ',
                'PC: ใช้ตรวจจับอนุภาคขนาดเล็กพิเศษที่อาจมองไม่เห็นด้วยการวัด PM เช่น ในพื้นที่ที่มีมลพิษสูง',
                'ทั้งสองช่วยในการกำหนดนโยบายควบคุมมลพิษและปกป้องสุขภาพประชาชน'
            ]
        }
    ]
};

const TriviaPopupContent = ({ onClose }) => {
    const [selectedTopic, setSelectedTopic] = useState(null);
    const popupRef = useRef(null);

    const topics = [
        { key: 'PM2.5', title: TRIVIA_DATA['PM2.5'].title },
        { key: 'PM10', title: TRIVIA_DATA['PM10'].title },
        { key: 'HourlyMeanPC0.1', title: TRIVIA_DATA['HourlyMeanPC0.1'].title },
        { key: 'DailyMeanPC0.1', title: TRIVIA_DATA['DailyMeanPC0.1'].title },
        { key: 'PCvsPM_0', title: TRIVIA_DATA['PCvsPM'][0].title },
        { key: 'PCvsPM_1', title: TRIVIA_DATA['PCvsPM'][1].title },
        { key: 'PCvsPM_2', title: TRIVIA_DATA['PCvsPM'][2].title }
    ];

    // Trap focus within popup for accessibility
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Escape') {
                if (selectedTopic) {
                    setSelectedTopic(null);
                } else {
                    onClose();
                }
            }
            if (e.key === 'Tab' && popupRef.current) {
                const focusableElements = popupRef.current.querySelectorAll(
                    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
                );
                const firstElement = focusableElements[0];
                const lastElement = focusableElements[focusableElements.length - 1];

                if (e.shiftKey && document.activeElement === firstElement) {
                    e.preventDefault();
                    lastElement.focus();
                } else if (!e.shiftKey && document.activeElement === lastElement) {
                    e.preventDefault();
                    firstElement.focus();
                }
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        if (popupRef.current) {
            popupRef.current.focus();
        }

        return () => {
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [selectedTopic, onClose]);

    const renderDetailPopup = (topicKey) => {
        let data;
        if (topicKey.startsWith('PCvsPM_')) {
            const index = parseInt(topicKey.split('_')[1]);
            data = TRIVIA_DATA['PCvsPM'][index];
        } else {
            data = TRIVIA_DATA[topicKey];
        }

        return (
            <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50 p-4">
                <div
                    ref={popupRef}
                    className="bg-white rounded-xl w-full max-w-md sm:max-w-lg md:max-w-xl mx-4 p-6 max-h-[85vh] overflow-y-auto shadow-2xl"
                    role="dialog"
                    aria-modal="true"
                    aria-labelledby="trivia-detail-title"
                    tabIndex={-1}
                >
                    {/* Header */}
                    <div className="flex justify-between items-start border-b border-gray-200 pb-3 mb-4">
                        <h3 id="trivia-detail-title" className="text-xl sm:text-2xl font-bold text-gray-800 font-sarabun leading-tight">
                            {data.title}
                        </h3>
                        <button
                            onClick={() => setSelectedTopic(null)}
                            className="text-gray-500 hover:text-gray-700 text-2xl font-bold transition-all duration-200 p-1 rounded-full hover:bg-gray-100"
                            aria-label="ย้อนกลับ"
                        >
                            ×
                        </button>
                    </div>

                    {/* Scrollable Content */}
                    <div className="space-y-5">
                        {data.description && typeof data.description === 'string' && (
                            <div className="flex items-start space-x-3">
                                <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                                    <svg className="w-5 h-5 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm-1-4a1 1 0 011-1h.01a1 1 0 110 2H10a1 1 0 01-1-1zm0-4a1 1 0 011-1h.01a1 1 0 110 2H10a1 1 0 01-1-1z" clipRule="evenodd" />
                                    </svg>
                                </div>
                                <p className="text-base sm:text-lg text-gray-700 font-sarabun leading-relaxed">{data.description}</p>
                            </div>
                        )}
                        {data.description && Array.isArray(data.description) && (
                            <div className="space-y-3">
                                {data.description.map((item, idx) => (
                                    <div key={idx} className="flex items-start space-x-3">
                                        <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                                            <svg className="w-5 h-5 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                                                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                            </svg>
                                        </div>
                                        <p className="text-base sm:text-lg text-gray-700 font-sarabun leading-relaxed">{item}</p>
                                    </div>
                                ))}
                            </div>
                        )}
                        {data.sources && (
                            <div className="flex items-start space-x-3">
                                <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                                    <svg className="w-5 h-5 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                                        <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
                                    </svg>
                                </div>
                                <div>
                                    <h4 className="text-base sm:text-lg font-bold text-gray-800 font-sarabun mb-2">แหล่งที่มา</h4>
                                    <p className="text-base sm:text-lg text-gray-700 font-sarabun leading-relaxed">{data.sources}</p>
                                </div>
                            </div>
                        )}
                        {data['short-term'] && (
                            <div className="flex items-start space-x-3">
                                <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                                    <svg className="w-5 h-5 text-orange-600" fill="currentColor" viewBox="0 0 20 20">
                                        <path fillRule="evenodd" d="M10 2a1 1 0 00-1 1v1a1 1 0 002 0V3a1 1 0 00-1-1zm0 12a1 1 0 100 2 1 1 0 000-2zm1-5a1 1 0 11-2 0 1 1 0 012 0z" clipRule="evenodd" />
                                    </svg>
                                </div>
                                <div>
                                    <h4 className="text-base sm:text-lg font-bold text-gray-800 font-sarabun mb-2">ผลกระทบระยะสั้น</h4>
                                    <div className="space-y-2">
                                        {data['short-term'].map((item, idx) => (
                                            <p key={idx} className="text-base sm:text-lg text-gray-700 font-sarabun leading-relaxed">• {item}</p>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}
                        {data['long-term'] && (
                            <div className="flex items-start space-x-3">
                                <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                                    <svg className="w-5 h-5 text-red-600" fill="currentColor" viewBox="0 0 20 20">
                                        <path fillRule="evenodd" d="M10 2a1 1 0 00-1 1v1a1 1 0 002 0V3a1 1 0 00-1-1zm0 12a1 1 0 100 2 1 1 0 000-2zm1-5a1 1 0 11-2 0 1 1 0 012 0z" clipRule="evenodd" />
                                    </svg>
                                </div>
                                <div>
                                    <h4 className="text-base sm:text-lg font-bold text-gray-800 font-sarabun mb-2">ผลกระทบระยะยาว</h4>
                                    <div className="space-y-2">
                                        {data['long-term'].map((item, idx) => (
                                            <p key={idx} className="text-base sm:text-lg text-gray-700 font-sarabun leading-relaxed">• {item}</p>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Footer Button */}
                    <div className="mt-6 pt-4 border-t border-gray-200">
                        <button
                            onClick={() => setSelectedTopic(null)}
                            className="w-full bg-green-500 hover:bg-green-600 text-white py-3 px-4 rounded-lg transition-all duration-200 text-lg font-semibold font-sarabun"
                        >
                            ย้อนกลับ
                        </button>
                    </div>
                </div>
            </div>
        );
    };

    // ถ้ามี selectedTopic แสดงแค่ detail popup
    if (selectedTopic) {
        return renderDetailPopup(selectedTopic);
    }

    // ถ้าไม่มี selectedTopic แสดงรายการหัวข้อ
    return (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50 p-4">
            <div
                ref={popupRef}
                className="bg-white rounded-xl w-full max-w-md sm:max-w-lg md:max-w-xl mx-4 p-6 max-h-[85vh] overflow-y-auto shadow-2xl"
                role="dialog"
                aria-modal="true"
                aria-labelledby="trivia-topics-title"
                tabIndex={-1}
            >
                {/* Header */}
                <div className="flex justify-between items-start border-b border-gray-200 pb-3 mb-4">
                    <h3 id="trivia-topics-title" className="text-xl sm:text-2xl font-bold text-gray-800 font-sarabun">เกร็ดความรู้</h3>
                    <button
                        onClick={onClose}
                        className="text-gray-500 hover:text-gray-700 text-2xl font-bold transition-all duration-200 p-1 rounded-full hover:bg-gray-100"
                        aria-label="ปิดป๊อปอัพ"
                    >
                        ×
                    </button>
                </div>

                {/* Description */}
                <div className="mb-4">
                    <div className="flex items-start space-x-3">
                        <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                            <svg className="w-5 h-5 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm-1-4a1 1 0 011-1h.01a1 1 0 110 2H10a1 1 0 01-1-1zm0-4a1 1 0 011-1h.01a1 1 0 110 2H10a1 1 0 01-1-1z" clipRule="evenodd" />
                            </svg>
                        </div>
                        <p className="text-base sm:text-lg text-gray-700 font-sarabun leading-relaxed">คลิกที่หัวข้อเพื่อดูรายละเอียดเกี่ยวกับมลพิษทางอากาศ</p>
                    </div>
                </div>

                {/* Scrollable Topics List */}
                <div className="space-y-3 max-h-[50vh] overflow-y-auto">
                    {topics.map((topic, index) => (
                        <button
                            key={topic.key}
                            onClick={() => setSelectedTopic(topic.key)}
                            className="w-full text-left text-base sm:text-lg text-gray-800 hover:text-blue-900 bg-gray-50 hover:bg-blue-50 font-sarabun py-3 px-4 rounded-lg border border-gray-200 hover:border-blue-300 transition-all duration-200"
                        >
                            <div className="flex items-center space-x-3">
                                <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center flex-shrink-0 border border-gray-300">
                                    <span className="text-sm font-semibold text-gray-700">{index + 1}</span>
                                </div>
                                <span className="leading-relaxed flex-1">{topic.title}</span>
                            </div>
                        </button>
                    ))}
                </div>

                {/* Footer Button */}
                <div className="mt-6 pt-4 border-t border-gray-200">
                    <button
                        onClick={onClose}
                        className="w-full bg-green-500 hover:bg-green-600 text-white py-3 px-4 rounded-lg transition-all duration-200 text-lg font-semibold font-sarabun"
                    >
                        ปิด
                    </button>
                </div>
            </div>
        </div>
    );
};

export { TRIVIA_DATA, TriviaPopupContent };