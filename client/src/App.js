import React, { useState } from 'react';
import axios from 'axios';
import { TextField, Button, Box, Typography } from '@mui/material';

const App = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState([]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const response = await axios.post('http://localhost:5000/api/predict', { text });
    setResult(response.data);
  }

  return (
    <Box sx={{ m: 2 }}>
      <form onSubmit={handleSubmit}>
        <TextField
          label="Input Text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          variant="outlined"
          fullWidth
        />
        <Button type="submit" variant="contained" color="primary" sx={{ mt: 2 }}>
          Submit
        </Button>
      </form>
      <Box sx={{ mt: 3 }}>
        {result.map((value, index) => (
          <Typography
            key={index}
            component="span"
            sx={{ color: value > 0 ? 'blue' : 'red' }}
          >
            {value}
          </Typography>
        ))}
      </Box>
    </Box>
  );
};

export default App;
