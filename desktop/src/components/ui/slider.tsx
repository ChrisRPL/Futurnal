/**
 * Slider Component
 *
 * Range slider input following Futurnal monochrome design system.
 * Uses native HTML5 range input with custom styling.
 */

import * as React from 'react';
import { cn } from '@/lib/utils';

export interface SliderProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'onChange' | 'type'> {
  /** Minimum value */
  min?: number;
  /** Maximum value */
  max?: number;
  /** Current value */
  value?: number;
  /** Step increment */
  step?: number;
  /** Change handler */
  onChange?: (value: number) => void;
}

const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  ({ className, min = 0, max = 100, step = 1, value, onChange, ...props }, ref) => {
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange?.(Number(e.target.value));
    };

    // Calculate fill percentage for gradient
    const percentage = ((value ?? min) - min) / (max - min) * 100;

    return (
      <input
        ref={ref}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={handleChange}
        className={cn(
          'w-full h-1.5 appearance-none cursor-pointer',
          'bg-white/10 rounded-full',
          '[&::-webkit-slider-thumb]:appearance-none',
          '[&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3',
          '[&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:rounded-full',
          '[&::-webkit-slider-thumb]:cursor-pointer',
          '[&::-webkit-slider-thumb]:hover:bg-white/90',
          '[&::-webkit-slider-thumb]:transition-colors',
          '[&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:h-3',
          '[&::-moz-range-thumb]:bg-white [&::-moz-range-thumb]:rounded-full',
          '[&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:cursor-pointer',
          'focus:outline-none focus-visible:ring-1 focus-visible:ring-white',
          className
        )}
        style={{
          background: `linear-gradient(to right, rgba(255,255,255,0.4) 0%, rgba(255,255,255,0.4) ${percentage}%, rgba(255,255,255,0.1) ${percentage}%, rgba(255,255,255,0.1) 100%)`,
        }}
        {...props}
      />
    );
  }
);
Slider.displayName = 'Slider';

export interface RangeSliderProps {
  /** Minimum value */
  min?: number;
  /** Maximum value */
  max?: number;
  /** Current range [min, max] */
  value?: [number, number];
  /** Step increment */
  step?: number;
  /** Change handler */
  onChange?: (value: [number, number]) => void;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Dual-thumb range slider for selecting a range of values.
 */
function RangeSlider({
  min = 0,
  max = 100,
  step = 1,
  value = [min, max],
  onChange,
  className,
}: RangeSliderProps) {
  const [minVal, maxVal] = value;

  const handleMinChange = (newMin: number) => {
    // Ensure min doesn't exceed max
    const clampedMin = Math.min(newMin, maxVal - step);
    onChange?.([clampedMin, maxVal]);
  };

  const handleMaxChange = (newMax: number) => {
    // Ensure max doesn't go below min
    const clampedMax = Math.max(newMax, minVal + step);
    onChange?.([minVal, clampedMax]);
  };

  // Calculate percentages for the range track
  const minPercent = ((minVal - min) / (max - min)) * 100;
  const maxPercent = ((maxVal - min) / (max - min)) * 100;

  return (
    <div className={cn('relative h-6', className)}>
      {/* Background track */}
      <div className="absolute top-1/2 -translate-y-1/2 w-full h-1.5 bg-white/10 rounded-full" />

      {/* Active range track */}
      <div
        className="absolute top-1/2 -translate-y-1/2 h-1.5 bg-white/40 rounded-full"
        style={{
          left: `${minPercent}%`,
          width: `${maxPercent - minPercent}%`,
        }}
      />

      {/* Min slider */}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={minVal}
        onChange={(e) => handleMinChange(Number(e.target.value))}
        className={cn(
          'absolute top-0 w-full h-6 appearance-none bg-transparent cursor-pointer pointer-events-none',
          '[&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:pointer-events-auto',
          '[&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3',
          '[&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:rounded-full',
          '[&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:z-10',
          '[&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:h-3',
          '[&::-moz-range-thumb]:bg-white [&::-moz-range-thumb]:rounded-full',
          '[&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:cursor-pointer',
          '[&::-moz-range-thumb]:pointer-events-auto'
        )}
      />

      {/* Max slider */}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={maxVal}
        onChange={(e) => handleMaxChange(Number(e.target.value))}
        className={cn(
          'absolute top-0 w-full h-6 appearance-none bg-transparent cursor-pointer pointer-events-none',
          '[&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:pointer-events-auto',
          '[&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3',
          '[&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:rounded-full',
          '[&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:z-10',
          '[&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:h-3',
          '[&::-moz-range-thumb]:bg-white [&::-moz-range-thumb]:rounded-full',
          '[&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:cursor-pointer',
          '[&::-moz-range-thumb]:pointer-events-auto'
        )}
      />
    </div>
  );
}

export { Slider, RangeSlider };
