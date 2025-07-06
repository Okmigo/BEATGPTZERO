const express = require('express');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 8080;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'build')));

app.post('/analyze', (req, res) => {
  const { text } = req.body;

  if (!text) {
    return res.status(400).json({ error: 'Missing text' });
  }

  const isHumanWritten = Math.random() > 0.5;

  res.json({ result: isHumanWritten ? 'Human' : 'AI', confidence: Math.random().toFixed(2) });
});

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.listen(PORT, () => {
  console.log(\`Server is running on port \${PORT}\`);
});
