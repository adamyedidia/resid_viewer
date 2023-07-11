import React, { useState } from 'react';
import { Select, MenuItem, Button } from '@mui/material';
export default function App() {
  const [type, setType] = useState('');
  const handleChange = (event) => {
    setType(event.target.value);
  };
  const onClick = () => {
    console.log('Hello!');
  };
  return (
    <>
      <Select value={type} onChange={handleChange}>
        <MenuItem value={10}>Type1</MenuItem>
        <MenuItem value={20}>Type2</MenuItem>
      </Select>
      <Button onClick={onClick}>Hello!</Button>
    </>
  );
}