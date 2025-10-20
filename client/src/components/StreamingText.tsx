"use client";

import { useEffect, useRef, useState } from 'react';

interface StreamingTextProps {
  text: string;
  className?: string;
  speed?: number;
  isComplete?: boolean;
  onComplete?: () => void;
}

export function StreamingText({ 
  text, className = '', speed = 50, isComplete = false, onComplete 
}: StreamingTextProps) {
  const [displayedText, setDisplayedText] = useState('');
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const indexRef = useRef(0);
  const lastTextRef = useRef('');

  useEffect(() => {
    if (text === lastTextRef.current) return;
    if (timeoutRef.current) { clearTimeout(timeoutRef.current); timeoutRef.current = null; }
    if (!text) { setDisplayedText(''); indexRef.current = 0; lastTextRef.current = text; return; }
    if (isComplete) { setDisplayedText(text); indexRef.current = text.length; lastTextRef.current = text; onComplete?.(); return; }
    if (text.length < lastTextRef.current.length || !text.startsWith(lastTextRef.current)) {
      indexRef.current = 0; setDisplayedText('');
    }
    const startIndex = indexRef.current;
    const animateText = () => {
      if (indexRef.current < text.length) {
        setDisplayedText(text.slice(0, indexRef.current + 1));
        indexRef.current++;
        const delay = 1000 / speed;
        timeoutRef.current = setTimeout(animateText, delay);
      } else {
        timeoutRef.current = null;
        if (isComplete) onComplete?.();
      }
    };
    if (startIndex < text.length) animateText();
    lastTextRef.current = text;
    return () => { if (timeoutRef.current) { clearTimeout(timeoutRef.current); timeoutRef.current = null; } };
  }, [text, speed, isComplete, onComplete]);

  return (
    <span className={className}>
      {displayedText}
      {!isComplete && displayedText.length > 0 && (<span className="animate-pulse opacity-70">|</span>)}
    </span>
  );
}

