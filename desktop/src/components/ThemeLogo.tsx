/**
 * ThemeLogo - Theme-aware logo component
 *
 * Automatically switches between light and dark logo variants
 * based on the current resolved theme.
 */

import { useTheme } from '@/contexts/ThemeContext';

type LogoVariant = 'small' | 'big' | 'horizontal';

interface ThemeLogoProps {
  /** Logo variant to display */
  variant?: LogoVariant;
  /** Alt text for the logo */
  alt?: string;
  /** Additional class names */
  className?: string;
}

const LOGO_PATHS: Record<LogoVariant, { light: string; dark: string }> = {
  small: {
    light: '/logo.png',
    dark: '/logo_dark.png',
  },
  big: {
    light: '/logo_big.png',
    dark: '/logo_big_dark.png',
  },
  horizontal: {
    light: '/logo_text_horizon.png',
    dark: '/logo_text_horizon_dark.png',
  },
};

export function ThemeLogo({
  variant = 'small',
  alt = 'Futurnal',
  className,
}: ThemeLogoProps) {
  const { resolvedTheme } = useTheme();

  const src = LOGO_PATHS[variant][resolvedTheme === 'light' ? 'light' : 'dark'];

  return <img src={src} alt={alt} className={className} />;
}
