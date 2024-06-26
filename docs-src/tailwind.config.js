/** @type {import("tailwindcss").Config} */
const colors = require("tailwindcss/colors");

module.exports = {
  mode: "jit",
  darkMode: "media",
  content: {
    files: ["*.html", "./src/**/*.rs"],
  },
  theme: {
    extend: {
      screens: {
        sm: "0px",
        md: "600px",
      },
    },
    colors: {
      lightbg: colors.stone,
      lighttext: colors.stone,
      lightaccent: colors.amber,
      darkbg: colors.gray,
      darktext: colors.gray,
      darkaccent: colors.blue,
    },
  },
  variants: {
    extend: {},
  },
  plugins: [],
};
