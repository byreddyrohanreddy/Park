/** @type {import('tailwindcss').Config} */

export default {
  darkMode: "class",

  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],

  theme: {
    extend: {
      colors: {
        primary: "#2563eb",
        secondary: "#14b8a6",
        medical: "#0f766e",
      },

      boxShadow: {
        card: "0 10px 30px rgba(0,0,0,0.08)",
      },
    },
  },

  plugins: [],
};