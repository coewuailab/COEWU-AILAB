// Animation.js
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';

// Hook สำหรับตรวจสอบ prefers-reduced-motion
const useReducedMotion = () => {
    const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

    useEffect(() => {
        const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
        setPrefersReducedMotion(mediaQuery.matches);

        const handleChange = (event) => setPrefersReducedMotion(event.matches);
        mediaQuery.addEventListener('change', handleChange);

        return () => mediaQuery.removeEventListener('change', handleChange);
    }, []);

    return prefersReducedMotion;
};

// Component สำหรับ Digit เดี่ยวที่มี delay
const CascadeDigit = React.memo(({ digit, className = "", delay = 0, animationType = "slide", duration = 500, onAnimationComplete }) => {
    const [currentDigit, setCurrentDigit] = useState(digit);
    const [isAnimating, setIsAnimating] = useState(false);
    const [shouldAnimate, setShouldAnimate] = useState(false);
    const prefersReducedMotion = useReducedMotion();
    const timeoutRefs = useRef([]);

    useEffect(() => {
        return () => {
            timeoutRefs.current.forEach(timeout => clearTimeout(timeout));
            timeoutRefs.current = [];
        };
    }, []);

    useEffect(() => {
        if (digit !== currentDigit) {
            if (prefersReducedMotion) {
                setCurrentDigit(digit);
                onAnimationComplete?.();
                return;
            }

            timeoutRefs.current.forEach(timeout => clearTimeout(timeout));
            timeoutRefs.current = [];

            const delayTimer = setTimeout(() => {
                setShouldAnimate(true);
                setIsAnimating(true);

                const animationTimer = setTimeout(() => {
                    setCurrentDigit(digit);
                    setIsAnimating(false);
                    setShouldAnimate(false);
                    onAnimationComplete?.();
                }, duration);

                timeoutRefs.current.push(animationTimer);
            }, delay);

            timeoutRefs.current.push(delayTimer);
        }
    }, [digit, currentDigit, delay, duration, prefersReducedMotion, onAnimationComplete]);

    const getAnimationClass = useCallback(() => {
        if (!shouldAnimate || prefersReducedMotion) return '';

        switch (animationType) {
            case "slide": return 'animate-cascade-slide-in';
            case "flip": return 'animate-cascade-flip-in';
            case "fade": return 'animate-cascade-fade-in';
            default: return '';
        }
    }, [shouldAnimate, animationType, prefersReducedMotion]);

    const getTransformStyle = useCallback(() => {
        if (prefersReducedMotion) return {};

        const easing = "cubic-bezier(0.25, 0.46, 0.45, 0.94)";

        switch (animationType) {
            case "slide":
                return {
                    transform: isAnimating ? 'translateY(-100%) scale(0.95)' : 'translateY(0) scale(1)',
                    opacity: isAnimating ? 0 : 1,
                    filter: isAnimating ? 'blur(1px)' : 'blur(0px)',
                    transition: `all ${duration}ms ${easing}`
                };
            case "flip":
                return {
                    transform: isAnimating ? 'rotateX(-90deg) scale(0.9)' : 'rotateX(0deg) scale(1)',
                    opacity: isAnimating ? 0 : 1,
                    transition: `all ${duration}ms ${easing}`,
                    transformOrigin: 'center bottom'
                };
            case "fade":
                return {
                    opacity: isAnimating ? 0 : 1,
                    transform: isAnimating ? 'scale(0.8)' : 'scale(1)',
                    filter: isAnimating ? 'blur(2px)' : 'blur(0px)',
                    transition: `all ${duration}ms ${easing}`
                };
            default:
                return {};
        }
    }, [isAnimating, animationType, duration, prefersReducedMotion]);

    return (
        <div className={`relative inline-block overflow-hidden transform-cascade ${className}`}>
            <span
                className={`font-numbers block transform-gpu ${getAnimationClass()}`}
                style={{
                    ...getTransformStyle(),
                    perspective: '1000px',
                    transformStyle: 'preserve-3d'
                }}
            >
                {shouldAnimate ? digit : currentDigit}
            </span>
        </div>
    );
});

CascadeDigit.displayName = 'CascadeDigit';

// Cascade Number Component
const CascadeNumber = React.memo(({ value, decimals = 3, className = "", animationType = "slide", direction = "left-to-right", staggerDelay = 80, duration = 500, showChangeIndicator = false, minDigits = 1, theme = "default" }) => {
    const [trend, setTrend] = useState("stable");
    const previousValue = useRef(value);
    const prefersReducedMotion = useReducedMotion();

    const formattedParts = useMemo(() => {
        try {
            const formattedValue = Number(value).toFixed(decimals);
            const [integerPart, decimalPart] = formattedValue.split('.');
            const paddedInteger = integerPart.padStart(minDigits, '0');

            return [
                ...paddedInteger.split('').map((d, i) => ({
                    type: 'integer',
                    digit: d,
                    index: i,
                    key: `int-${i}`
                })),
                ...(decimals > 0 ? [{
                    type: 'decimal-point',
                    digit: '.',
                    index: 'dot',
                    key: 'dot'
                }] : []),
                ...(decimals > 0 ? decimalPart.split('').map((d, i) => ({
                    type: 'decimal',
                    digit: d,
                    index: i,
                    key: `dec-${i}`
                })) : [])
            ];
        } catch (error) {
            console.warn('CascadeNumber: Invalid value provided', value);
            return [];
        }
    }, [value, decimals, minDigits]);

    useEffect(() => {
        if (previousValue.current < value) {
            setTrend("up");
        } else if (previousValue.current > value) {
            setTrend("down");
        } else {
            setTrend("stable");
        }

        const timer = setTimeout(() => setTrend("stable"), prefersReducedMotion ? 500 : 2500);
        previousValue.current = value;

        return () => clearTimeout(timer);
    }, [value, prefersReducedMotion]);

    const getDigitDelay = useCallback((index) => {
        if (prefersReducedMotion) return 0;

        if (direction === "right-to-left") {
            return (formattedParts.length - 1 - index) * staggerDelay;
        }
        return index * staggerDelay;
    }, [direction, formattedParts.length, staggerDelay, prefersReducedMotion]);

    const getTrendStyles = useCallback(() => {
        const baseClasses = "text-xs transition-all duration-700 ease-out ml-2 transform";

        switch (trend) {
            case "up":
                return {
                    className: `${baseClasses} text-green-400`,
                    style: {
                        transform: 'translateY(-3px)',
                        opacity: 1
                    },
                    icon: ""
                };
            case "down":
                return {
                    className: `${baseClasses} text-red-400`,
                    style: {
                        transform: 'translateY(3px)',
                        opacity: 1
                    },
                    icon: ""
                };
            default:
                return {
                    className: `${baseClasses} text-gray-400`,
                    style: {
                        transform: 'translateY(0)',
                        opacity: 0
                    },
                    icon: ""
                };
        }
    }, [trend]);

    const trendStyles = getTrendStyles();

    return (
        <div className={`relative inline-flex items-center gap-0 cascade-container ${theme}`}>
            <div className={`inline-flex ${className}`}>
                {formattedParts.map((part, index) => (
                    <div
                        key={part.key}
                        className="cascade-digit-wrapper"
                        style={{ minWidth: '0.6em' }}
                    >
                        {part.digit === '.' ? (
                            <span className="font-numbers">.</span>
                        ) : (
                            <CascadeDigit
                                digit={part.digit}
                                className="text-center"
                                delay={getDigitDelay(index)}
                                animationType={animationType}
                                duration={duration}
                            />
                        )}
                    </div>
                ))}
            </div>

            {showChangeIndicator && trend !== "stable" && (
                <span
                    className={trendStyles.className}
                    style={{
                        ...trendStyles.style,
                        animation: !prefersReducedMotion ? 'cascadeFadeInOut 3s cubic-bezier(0.25, 0.46, 0.45, 0.94)' : 'none'
                    }}
                    aria-live="polite"
                    aria-label={`Value ${trend}`}
                >
                    {trendStyles.icon}
                </span>
            )}
        </div>
    );
});

CascadeNumber.displayName = 'CascadeNumber';

// Wave Effect Number Component
const WaveNumber = React.memo(({ value, decimals = 3, className = "", waveDelay = 60, duration = 600 }) => {
    const [displayDigits, setDisplayDigits] = useState([]);
    const [isWaving, setIsWaving] = useState(false);
    const previousValue = useRef(value);
    const prefersReducedMotion = useReducedMotion();
    const timeoutRefs = useRef([]);

    useEffect(() => {
        return () => {
            timeoutRefs.current.forEach(timeout => clearTimeout(timeout));
            timeoutRefs.current = [];
        };
    }, []);

    useEffect(() => {
        try {
            const formattedValue = Number(value).toFixed(decimals);
            const [integerPart, decimalPart] = formattedValue.split('.');
            const newDigits = [...integerPart.split(''), '.', ...decimalPart.split('')];

            if (previousValue.current !== value) {
                setIsWaving(true);

                timeoutRefs.current.forEach(timeout => clearTimeout(timeout));
                timeoutRefs.current = [];

                if (prefersReducedMotion) {
                    setDisplayDigits(newDigits.map(digit => ({ digit, isNew: false })));
                    setIsWaving(false);
                } else {
                    newDigits.forEach((digit, index) => {
                        const timeout = setTimeout(() => {
                            setDisplayDigits(prev => {
                                const updated = [...prev];
                                updated[index] = {
                                    digit,
                                    isNew: true,
                                    timestamp: Date.now()
                                };
                                return updated;
                            });
                        }, index * waveDelay);

                        timeoutRefs.current.push(timeout);
                    });

                    const resetTimeout = setTimeout(() => {
                        setIsWaving(false);
                        setDisplayDigits(newDigits.map(digit => ({ digit, isNew: false })));
                    }, newDigits.length * waveDelay + duration);

                    timeoutRefs.current.push(resetTimeout);
                }

                previousValue.current = value;
            } else if (displayDigits.length === 0) {
                setDisplayDigits(newDigits.map(digit => ({ digit, isNew: false })));
            }
        } catch (error) {
            console.warn('WaveNumber: Invalid value provided', value);
        }
    }, [value, decimals, waveDelay, duration, prefersReducedMotion]);

    return (
        <div className={`inline-flex wave-effect ${className}`}>
            {displayDigits.map((item, index) => (
                <div
                    key={index}
                    className="relative overflow-hidden"
                    style={{ minWidth: item.digit === '.' ? '4px' : '0.6em' }}
                >
                    <span
                        className={`
              font-numbers block text-center transform-gpu transition-all ease-out
              ${item.isNew && !prefersReducedMotion ? 'animate-wave-bounce' : ''}
            `}
                        style={{
                            transform: item.isNew && !prefersReducedMotion
                                ? 'translateY(-10px) scale(1.1)'
                                : 'translateY(0) scale(1)',
                            transition: `all ${duration}ms cubic-bezier(0.25, 0.46, 0.45, 0.94)`,
                            transitionDelay: prefersReducedMotion ? '0ms' : `${index * waveDelay}ms`,
                            filter: item.isNew && !prefersReducedMotion
                                ? 'drop-shadow(0 0 8px rgba(59, 130, 246, 0.5))'
                                : 'none'
                        }}
                    >
                        {item.digit}
                    </span>
                </div>
            ))}
        </div>
    );
});

WaveNumber.displayName = 'WaveNumber';

// Smooth Ripple Effect
const RippleNumber = React.memo(({ value, decimals = 3, className = "", rippleDelay = 50, duration = 800 }) => {
    const [animatingDigits, setAnimatingDigits] = useState(new Set());
    const previousValue = useRef(value);
    const prefersReducedMotion = useReducedMotion();
    const timeoutRefs = useRef([]);

    useEffect(() => {
        return () => {
            timeoutRefs.current.forEach(timeout => clearTimeout(timeout));
            timeoutRefs.current = [];
        };
    }, []);

    useEffect(() => {
        if (previousValue.current !== value) {
            try {
                const formattedValue = Number(value).toFixed(decimals);
                const [integerPart, decimalPart] = formattedValue.split('.');
                const allDigits = [...integerPart.split(''), '.', ...decimalPart.split('')];

                timeoutRefs.current.forEach(timeout => clearTimeout(timeout));
                timeoutRefs.current = [];

                if (prefersReducedMotion) {
                    previousValue.current = value;
                    return;
                }

                const center = Math.floor(allDigits.length / 2);

                allDigits.forEach((_, index) => {
                    const distance = Math.abs(index - center);
                    const delay = distance * rippleDelay;

                    const startTimeout = setTimeout(() => {
                        setAnimatingDigits(prev => new Set([...prev, index]));

                        const endTimeout = setTimeout(() => {
                            setAnimatingDigits(prev => {
                                const newSet = new Set(prev);
                                newSet.delete(index);
                                return newSet;
                            });
                        }, duration);

                        timeoutRefs.current.push(endTimeout);
                    }, delay);

                    timeoutRefs.current.push(startTimeout);
                });

                previousValue.current = value;
            } catch (error) {
                console.warn('RippleNumber: Invalid value provided', value);
            }
        }
    }, [value, decimals, rippleDelay, duration, prefersReducedMotion]);

    const formattedValue = useMemo(() => {
        try {
            return Number(value).toFixed(decimals);
        } catch {
            return '0'.repeat(decimals + 2);
        }
    }, [value, decimals]);

    const [integerPart, decimalPart] = formattedValue.split('.');
    const allDigits = [...integerPart.split(''), '.', ...decimalPart.split('')];

    return (
        <div className={`inline-flex ripple-effect ${className}`}>
            {allDigits.map((digit, index) => (
                <div
                    key={index}
                    className={`relative overflow-hidden ripple-digit ${animatingDigits.has(index) ? 'rippling' : ''}`}
                    style={{ minWidth: digit === '.' ? '4px' : '0.6em' }}
                >
                    <span
                        className={`
              font-numbers block text-center transform-gpu transition-all ease-out
              ${animatingDigits.has(index) && !prefersReducedMotion ? 'animate-ripple-glow' : ''}
            `}
                        style={{
                            transform: animatingDigits.has(index) && !prefersReducedMotion
                                ? 'translateY(-8px) scale(1.15)'
                                : 'translateY(0) scale(1)',
                            transition: `all ${duration}ms cubic-bezier(0.25, 0.46, 0.45, 0.94)`,
                            filter: animatingDigits.has(index) && !prefersReducedMotion
                                ? 'drop-shadow(0 4px 12px rgba(59, 130, 246, 0.4))'
                                : 'none',
                            textShadow: animatingDigits.has(index) && !prefersReducedMotion
                                ? '0 0 10px rgba(59, 130, 246, 0.6)'
                                : 'none'
                        }}
                    >
                        {digit}
                    </span>
                </div>
            ))}
        </div>
    );
});

RippleNumber.displayName = 'RippleNumber';

// Updated LiveActivityNumber
const LiveActivityNumber = React.memo(({ value, type = "cascade-slide", decimals = 3, className = "", showChangeIndicator = true, minDigits = 1, direction = "right-to-left", theme = "default", onValueChange, errorFallback = "---" }) => {
    const safeValue = useMemo(() => {
        if (typeof value !== 'number' || !isFinite(value)) {
            console.warn('LiveActivityNumber: Invalid value provided, using fallback');
            return 0;
        }
        return value;
    }, [value]);

    useEffect(() => {
        onValueChange?.(safeValue);
    }, [safeValue, onValueChange]);

    const commonProps = {
        value: safeValue,
        decimals,
        className,
        minDigits
    };

    try {
        switch (type) {
            case "cascade-slide":
                return (
                    <CascadeNumber
                        {...commonProps}
                        animationType="slide"
                        direction={direction}
                        staggerDelay={80}
                        duration={500}
                        showChangeIndicator={showChangeIndicator}
                        theme={theme}
                    />
                );
            case "cascade-flip":
                return (
                    <CascadeNumber
                        {...commonProps}
                        animationType="flip"
                        direction={direction}
                        staggerDelay={100}
                        duration={600}
                        showChangeIndicator={showChangeIndicator}
                        theme={theme}
                    />
                );
            case "cascade-fade":
                return (
                    <CascadeNumber
                        {...commonProps}
                        animationType="fade"
                        direction={direction}
                        staggerDelay={70}
                        duration={450}
                        showChangeIndicator={showChangeIndicator}
                        theme={theme}
                    />
                );
            case "wave":
                return (
                    <WaveNumber
                        {...commonProps}
                        waveDelay={60}
                        duration={600}
                    />
                );
            case "ripple":
                return (
                    <RippleNumber
                        {...commonProps}
                        rippleDelay={50}
                        duration={800}
                    />
                );
            default:
                return <span className={`font-numbers ${className}`}>{safeValue.toFixed(decimals)}</span>;
        }
    } catch (error) {
        console.error('LiveActivityNumber: Render error', error);
        return <span className={`font-numbers ${className}`}>{errorFallback}</span>;
    }
});

LiveActivityNumber.displayName = 'LiveActivityNumber';

export {
    CascadeDigit,
    CascadeNumber,
    WaveNumber,
    RippleNumber,
    LiveActivityNumber,
    useReducedMotion
};