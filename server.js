import express from 'express';
const app = express();

app.use(express.json());

app.post('/analyze', (req, res) => {
  // your processing logic
  res.json({ result: 'Hello from Cloud Run' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
