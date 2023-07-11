import React, { useState } from 'react';
import { Button, FormControl, InputLabel, MenuItem, Select } from '@mui/material';
import axios from 'axios';

const TypeSelector = ({ types, selectedType, onTypeChange }) => (
  <FormControl variant="filled">
    <InputLabel>Type</InputLabel>
    <Select
      value={selectedType}
      onChange={(event) => onTypeChange(event.target.value)}
    >
      {types.map((type) => <MenuItem value={type}>{type}</MenuItem>)}
    </Select>
  </FormControl>
);

const IndexSelector = ({ range, selectedIndex, onIndexChange }) => (
  <FormControl variant="filled">
    <InputLabel>Index</InputLabel>
    <Select
      value={selectedIndex}
      onChange={(event) => onIndexChange(event.target.value)}
    >
      {[...Array(range)].map((_, index) => <MenuItem value={index}>{index}</MenuItem>)}
    </Select>
  </FormControl>
);

const App = () => {
  const [selectedType, setSelectedType] = useState("");
  const [selectedHead, setSelectedHead] = useState("");
  const [selectedComponentIndex, setSelectedComponentIndex] = useState("");
  
  const types = ["Type1", "Type2"]; // Replace with actual types

  const fetchResids = async () => {
    try {
      const response = await axios.get("/api/resids", {
        params: {
          type: selectedType,
          head: selectedHead,
          component_index: selectedComponentIndex,
        },
      });
      // handle the response
      console.log(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <TypeSelector
        types={types}
        selectedType={selectedType}
        onTypeChange={(type) => setSelectedType(type)}
      />

      {/* Only show the Head and ComponentIndex selectors for certain types */}
      {selectedType === "Type1" && (
        <>
          <IndexSelector
            range={12}
            selectedIndex={selectedHead}
            onIndexChange={(index) => setSelectedHead(index)}
          />
          <IndexSelector
            range={64}
            selectedIndex={selectedComponentIndex}
            onIndexChange={(index) => setSelectedComponentIndex(index)}
          />
        </>
      )}

      <Button variant="contained" color="primary" onClick={fetchResids}>
        Submit
      </Button>
    </div>
  );
};

export default App;