require("dotenv").config();
const express = require("express");
const axios = require("axios");
const app = express();
const port = parseInt(process.env.PORT) || 8080;
const bodyParser = require("body-parser");
const multer = require("multer");
const upload = multer();
const path = require("path");

let text;
let resp;
let time;

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

app.get("/test", async (req, res) => {
  const preeverfinal = await rewrite("This is a sample test string for rewrite function.");
  res.send(preeverfinal);
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
  text = posttext;
  const preeverfinal = await rewrite(text);
  res.send(preeverfinal);
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
  console.log(`Example app listening at http://localhost:${port}`);
});

async function rewrite(text) {
  console.log("Starting rewrite for:", text);
  const start = Date.now();

  try {
    const first = await axios.post("https://api.spinbot.com/spin/rewrite-text", {
      text: text,
      x_spin_cap_words: false,
    }, {
      headers: {
        Origin: "https://spinbot.com",
        "Content-Type": "application/json",
      }
    });

    const resp1 = first.data;
    console.log("First Spinbot response:", resp1);

    const second = await axios.post("https://backend-spinbot-wrapper-prod.azurewebsites.net/spin/rewrite-text", {
      text: resp1,
      x_spin_cap_words: false,
    }, {
      headers: {
        Origin: "https://free-article-spinner.com",
        "Content-Type": "application/json",
      }
    });

    const resp2 = second.data;
    const time = ((Date.now() - start) / 1000).toFixed(2) + "s";
    console.log("Final rewrite:", resp2, "Time taken:", time);

    return { text: resp2, time };
  } catch (error) {
    console.error("Rewrite failed:", error.message);
    return { text: null, time: null };
  }
}
