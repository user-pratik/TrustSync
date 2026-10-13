/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        bg: 'rgb(var(--bg) / <alpha-value>)',
        elev: 'rgb(var(--elev) / <alpha-value>)',
        card: 'rgb(var(--card) / <alpha-value>)',
        border: 'rgb(var(--border) / <alpha-value>)',
        text: 'rgb(var(--text) / <alpha-value>)',
        muted: 'rgb(var(--muted) / <alpha-value>)',
        accept: 'rgb(var(--accept) / <alpha-value>)',
        reject: 'rgb(var(--reject) / <alpha-value>)',
        listen: 'rgb(var(--listen) / <alpha-value>)',
        info: 'rgb(var(--info) / <alpha-value>)',
        idle: 'rgb(var(--idle) / <alpha-value>)',
      },
      fontFamily: {
        sans: ['Inter Variable', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      borderRadius: {
        card: '16px',
      },
      boxShadow: {
        glass: '0 8px 30px rgba(0,0,0,.35)',
      },
    },
  },
  plugins: [],
}
