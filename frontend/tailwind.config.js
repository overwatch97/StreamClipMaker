/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#09090b", // very dark modern bg
        surface: "#18181b", // slightly lighter
        primary: "#3b82f6",
        accent: "#8b5cf6",
        success: "#22c55e",
        danger: "#ef4444",
        warning: "#eab308",
      }
    },
  },
  plugins: [],
}
