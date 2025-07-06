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
let resp1;

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
  const preeverfinal = await rewrite(
    "Human behavior is an incredibly complex and multifaceted subject..."
  );
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
    console.log("REWRITTEN TEXT:", rewrittenResult);

    res.json({
      original: text,
      rewritten: rewrittenResult?.text || "[Rewrite failed]",
      bypassable: !!rewrittenResult?.text
    });
  } catch (err) {
    console.error("Error in /analyze:", err.message);
    res.status(500).json({ error: "Internal server error", details: err.message });
  }
});


app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`);
});

async function rewrite(text) {
  console.log("started!");

  const options = {
    method: "POST",
    url: "https://api.spinbot.com/spin/rewrite-text",
    headers: {
      Origin: "https://spinbot.com",
      "Content-Type": "application/json",
    },
    data: {
      text: text,
      x_spin_cap_words: false,
    },
  };

  const start = Date.now();
  await axios
    .request(options)
    .then(async function (response) {
      resp1 = response.data;

      const options1 = {
        method: "POST",
        url: "https://backend-spinbot-wrapper-prod.azurewebsites.net/spin/rewrite-text",
        headers: {
          Origin: "https://free-article-spinner.com",
          "Content-Type": "application/json",
        },
        data: {
          text: resp1,
          x_spin_cap_words: false,
        },
      };

      await axios
        .request(options1)
        .then(function (newresponse) {
          const finish = Date.now();
          time = (finish - start) / 1000 + "s";
          resp = newresponse.data;
        })
        .catch(function (error) {
          console.error(error);
        });
    })
    .catch(async function (error) {
      console.error(error);
    });

  return { text: resp, time: time };
}
