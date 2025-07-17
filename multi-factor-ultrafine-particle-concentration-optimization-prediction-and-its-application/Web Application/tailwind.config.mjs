/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],

  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
      },
      fontFamily: {
        montserrat: ['var(--font-montserrat)', 'Montserrat', 'sans-serif'],
        sarabun: ['var(--font-sarabun)', 'Sarabun', 'sans-serif'],
        numbers: ['var(--font-numbers)', 'Rubik', 'sans-serif'],
        rubik: ['var(--font-numbers)', 'Rubik', 'sans-serif'], // alias
      },
    },
  },
  plugins: [],
};