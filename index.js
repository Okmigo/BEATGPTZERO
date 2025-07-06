require("dotenv").config();
const express = require("express");
const axios = require("axios");
const bodyParser = require("body-parser");
const multer = require("multer");
const path = require("path");

const app = express();
const port = parseInt(process.env.PORT) || 8080;
const upload = multer();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(upload.array());
app.use(express.static("public"));

app.get("/", (req, res) => res.sendFile(path.join(__dirname, "/views/index.html")));
app.get("/about", (req, res) => res.sendFile(path.join(__dirname, "/views/about.html")));
app.get("/style.css", (req, res) => res.sendFile(path.join(__dirname, "/views/style.css")));
app.get("/font", (req, res) => res.sendFile(path.join(__dirname, "/views/fonts/boxy-bold.ttf")));
app.get("/favicon.ico", (req, res) => res.sendFile(path.join(__dirname, "/views/favicon.ico")));
app.get("/logo.png", (req, res) => res.sendFile(path.join(__dirname, "/views/logo.png")));

app.get("/test", async (req, res) => {
  const rewritten = await rewrite("This is a sample test string.");
  res.json(rewritten);
});

app.post("/api", async (req, res) => {
  const input = req.body.text?.replace(/(\r\n|\n|\r)/gm, "") || "";
  const rewritten = await rewrite(input);
  res.json(rewritten);
});

app.post("/analyze", async (req, res) => {
  const { text } = req.body;

  if (!text || text.trim() === "") {
    return res.status(400).json({ error: "Missing 'text' in request body." });
  }

  const result = await rewrite(text);

  if (!result.text) {
    return res.status(500).json({
      original: text,
      rewritten: null,
      bypassable: false,
      error: "Rewrite failed or returned no result",
    });
  }

  res.json({
    original: text,
    rewritten: result.text,
    bypassable: true,
    time: result.time,
  });
});

app.listen(port, () => {
  console.log(`✅ Server running at http://localhost:${port}`);
});

// 🌀 Rewrite logic using chained spinbot APIs
async function rewrite(inputText) {
  const start = Date.now();

  try {
    const spin1 = await axios.post(
      "https://api.spinbot.com/spin/rewrite-text",
      { text: inputText, x_spin_cap_words: false },
      {
        headers: {
          Origin: "https://spinbot.com",
          "Content-Type": "application/json",
        },
      }
    );

    const spin2 = await axios.post(
      "https://backend-spinbot-wrapper-prod.azurewebsites.net/spin/rewrite-text",
      { text: spin1.data, x_spin_cap_words: false },
      {
        headers: {
          Origin: "https://free-article-spinner.com",
          "Content-Type": "application/json",
        },
      }
    );

    const duration = ((Date.now() - start) / 1000).toFixed(2) + "s";
    return { text: spin2.data, time: duration };
  } catch (err) {
    console.error("❌ Rewrite failed:", err.message);
    return { text: null, time: null };
  }
}
