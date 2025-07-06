// index.js
require("dotenv").config();
const express = require("express");
const bodyParser = require("body-parser");
const multer = require("multer");
const path = require("path");
const { spawn } = require("child_process");

const app = express();
const port = parseInt(process.env.PORT) || 8080;

const upload = multer();
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(upload.array());
app.use(express.static("public"));

app.get("/style.css", (req, res) => {
  res.sendFile(path.join(__dirname, "/views/style.css"));
});

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "/views/index.html"));
});

app.get("/about", (req, res) => {
  res.sendFile(path.join(__dirname, "/views/about.html"));
});

app.get("/font", (req, res) => {
  res.sendFile(path.join(__dirname, "/views/fonts/boxy-bold.ttf"));
});

app.get("/favicon.ico", (req, res) => {
  res.sendFile(path.join(__dirname, "/views/favicon.ico"));
});

app.get("/logo.png", (req, res) => {
  res.sendFile(path.join(__dirname, "/views/logo.png"));
});

app.post("/api", async (req, res) => {
  const pretext = req.body.text;
  const posttext = pretext.replace(/(\r\n|\n|\r)/gm, "");
  const result = await rewrite(posttext);
  res.send(result);
});

app.post("/analyze", async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: "Missing 'text' in body" });

    const rewrittenResult = await rewrite(text);

    res.json({
      original: text,
      rewritten: rewrittenResult.text,
      bypassable: !!rewrittenResult.text
    });
  } catch (err) {
    res.status(500).json({ error: "Internal server error", details: err.message });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

// Local rewrite using a child process to execute Python logic (free and private)
async function rewrite(text) {
  return new Promise((resolve) => {
    const py = spawn("python3", ["./rewriter.py", text]);
    let data = "";

    py.stdout.on("data", (chunk) => {
      data += chunk.toString();
    });

    py.stderr.on("data", (err) => {
      console.error("Python error:", err.toString());
    });

    py.on("close", () => {
      resolve({ text: data.trim(), time: null });
    });
  });
}
