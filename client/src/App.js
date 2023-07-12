import React, { useState, useEffect } from 'react';
import { Button, Box, FormControl, InputLabel, MenuItem, Select, Grid, Paper } from '@mui/material';
import axios from 'axios';
import { gpt2_types } from './gpt2_types';
import { Typography } from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import chroma from 'chroma-js';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
  components: {
    MuiMenuItem: {
      styleOverrides: {
        root: {
          "&:hover": {
            backgroundColor: '#3a3a3a', // background color when hovered
          },
          "&.Mui-selected": {
            backgroundColor: '#3a3a3a', // background color when selected
          },
          "&.Mui-selected:hover": {
            backgroundColor: '#3a3a3a', // background color when selected and hovered
          },
        },
      },
    },
  },
});

const TypeSelector = ({ types, selectedType, onTypeChange }) => (
  <Grid item xs={12} md={4}>
    <FormControl variant="filled" fullWidth style={{backgroundColor: '#3a3a3a'}}>
      <InputLabel style={{ color: 'white' }}>Type</InputLabel>
      <Select
        value={selectedType}
        onChange={(event) => onTypeChange(event.target.value)}
      >
        {types.map((type) => <MenuItem value={type} key={type}>
        <Typography style={{ color: 'white' }}>{type}</Typography>
        </MenuItem>)}
      </Select>
    </FormControl>
  </Grid>
);

const IndexSelector = ({ range, selectedIndex, onIndexChange, label }) => (
  <Grid item xs={12} md={4}>
    <FormControl variant="filled" fullWidth style={{backgroundColor: '#3a3a3a'}}>
      <InputLabel style={{ color: 'white' }}>{label}</InputLabel>
      <Select
        value={selectedIndex}
        onChange={(event) => onIndexChange(event.target.value)}
      >
        {[...Array(range)].map((_, index) => <MenuItem value={index} key={index}>
          <Typography style={{ color: 'white' }}>{index}</Typography>
        </MenuItem>)}
      </Select>
    </FormControl>
  </Grid>
);

const ColoredResidBox = ({ resid, minDotProduct, maxDotProduct }) => {
  const { dotProduct } = resid;
  const normalizedDotProduct = (dotProduct - minDotProduct) / (maxDotProduct - minDotProduct);
  const colorScale = chroma.scale(['red', 'white', 'blue']).mode('lch');
  const color = colorScale(normalizedDotProduct).hex();
  const textColor = chroma.contrast(color, 'white') > chroma.contrast(color, 'black') ? 'white' : 'black';

  return (
    <Box px={0.1} py={1} m={0} bgcolor={color} color={textColor} component={Paper} sx={{borderRadius: 0, border: '1px solid black'}}>
      <Typography variant="body1"><pre style={{fontFamily: 'inherit', margin: 0}}>{resid.decodedToken}</pre></Typography>
    </Box>
  );
};

const PromptRow = ({ promptId, resids, maxDotProduct, minDotProduct }) => (
  <Box display="flex" flexWrap="wrap">
    {resids.map((resid) => <ColoredResidBox key={resid.id} resid={resid} maxDotProduct={maxDotProduct} minDotProduct={minDotProduct} />)}
  </Box>
);

const App = () => {
  const [selectedType, setSelectedType] = useState("");
  const [selectedHead, setSelectedHead] = useState("");
  const [selectedComponentIndex, setSelectedComponentIndex] = useState("");
  const [resids, setResids] = useState([]);
  const [direction, setDirection] = useState([]);
  const [maxDotProduct, setMaxDotProduct] = useState(1);
  const [minDotProduct, setMinDotProduct] = useState(-1);

  const calculateDotProducts = (residsToCalculate, directionToCalculate) => {
    if (!residsToCalculate?.length || !directionToCalculate?.direction) return;
    console.log('calculating dot products')
    const dotProducts = residsToCalculate.map(resid => {
      const dotProduct = resid.resid.reduce((sum, value, i) => sum + value * directionToCalculate.direction[i], 0);
      return {...resid, dotProduct};
    });
    const maxDotProduct = Math.max(...dotProducts.map(({dotProduct}) => dotProduct));
    const minDotProduct = Math.min(...dotProducts.map(({dotProduct}) => dotProduct));
    setResids(dotProducts);
    setMaxDotProduct(maxDotProduct);
    setMinDotProduct(minDotProduct);
  }

  useEffect(() => {
    if (resids?.length) return;
    console.log('Fetching resids!')
    const fetchResids = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:5000/api/resids", {
          params: {
            model_name: "gpt2-small",
            type: selectedType,
            head: selectedHead,
            component_index: selectedComponentIndex,
          },
        });
        const newResids = response.data;
        calculateDotProducts(newResids, direction);
        setResids(newResids);
      } catch (error) {
        console.error(error);
      }
    };

    fetchResids();
  }, [selectedType, selectedHead, selectedComponentIndex, resids?.length]);

  useEffect(() => {
    console.log('Fetching direction!')
    const fetchDirection = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:5000/api/directions", {
          params: {
            model_name: "gpt2-small",
            type: selectedType,
            head: selectedHead,
            component_index: selectedComponentIndex,
          },
        });
        const newDirection = response.data;
        calculateDotProducts(resids, newDirection);
        setDirection(newDirection);
      } catch (error) {
        console.error(error);
      }
    };

    fetchDirection();
  }, [selectedType, selectedHead, selectedComponentIndex]);

  // useEffect(() => {
  //   if (!resids?.length || !direction?.direction) return;
  //   console.log('calculating dot products')
  //   const dotProducts = resids.map(resid => {
  //     const dotProduct = resid.resid.reduce((sum, value, i) => sum + value * direction.direction[i], 0);
  //     return {...resid, dotProduct};
  //   });
  //   const maxDotProduct = Math.max(...dotProducts.map(({dotProduct}) => dotProduct));
  //   const minDotProduct = Math.min(...dotProducts.map(({dotProduct}) => dotProduct));
  //   setResids(dotProducts);
  //   setMaxDotProduct(maxDotProduct);
  //   setMinDotProduct(minDotProduct);
  // }, [selectedType, selectedHead, selectedComponentIndex, resids?.length]);

  console.log(resids);
  console.log(direction);

  const groupedResids = resids.reduce((groups, resid) => {
    (groups[resid.promptId] = groups[resid.promptId] || []).push(resid);
    return groups;
  }, {});


  const appStyle = {
    backgroundColor: '#1b1b1b',
    minHeight: '100vh',
    color: '#f5f5f5',
    padding: '20px',
  };

  const types = (
    Object.keys(gpt2_types)
    .filter((type => !type.includes("hook_scale")))
    .filter((type => !type.includes("hook_attn_scores")))
    .filter((type => !type.includes("hook_pattern")))
  );
  const typeShape = gpt2_types[selectedType];

  const needsHead = !!typeShape?.includes(12);
  const maxComponentIndex = (
    typeShape?.includes(64) ? 64 : 
    typeShape?.includes(768) ? 768 :
    typeShape?.includes(3072) ? 3072 :
    null
  )

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
    <ThemeProvider theme={darkTheme}>
      <div style={appStyle}>
        <Grid container spacing={3} justify="center">
          <TypeSelector
            types={types}
            selectedType={selectedType}
            onTypeChange={(type) => setSelectedType(type)}
          />

          {needsHead && (
            <IndexSelector
              range={12}
              selectedIndex={selectedHead}
              onIndexChange={(index) => setSelectedHead(index)}
              label='Head'
            />
          )}
          {maxComponentIndex && (
            <IndexSelector
              // range={maxComponentIndex}
              range={30}
              selectedIndex={selectedComponentIndex}
              onIndexChange={(index) => setSelectedComponentIndex(index)}
              label='Component Index'
            />
          )}

          <Grid item xs={12}>
            <Box display="flex" justifyContent="flex-start">
              <Button variant="contained" color="primary" onClick={fetchResids}>
                Submit
              </Button>
            </Box>
          </Grid>
        </Grid>
        <Grid container spacing={1}>
          {Object.entries(groupedResids).map(([promptId, resids]) => (
            <Grid item xs={12} key={promptId}>
              <PromptRow promptId={promptId} resids={resids} maxDotProduct={maxDotProduct} minDotProduct={minDotProduct} />
            </Grid>
          ))}
        </Grid>
      </div>
    </ThemeProvider>
  );
};

export default App;
