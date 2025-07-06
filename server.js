const express = require('express');
const app = express();

app.use(express.json());

app.post('/analyze', (req, res) => {
  const text = req.body.text;
  // Temporary placeholder response
  res.json({ result: `Processed text: ${text}` });
});

// ✅ Cloud Run requires using the provided PORT env variable
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
