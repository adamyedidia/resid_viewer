import React, { useState, useEffect, useCallback } from 'react';
import { Button, Box, FormControl, InputLabel, MenuItem, Select, Grid, Paper, TextField, Slider } from '@mui/material';
import axios from 'axios';
import { gpt2_types } from './gpt2_types';
import { Typography } from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import chroma from 'chroma-js';
import Draggable from 'react-draggable';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import Dialog from '@mui/material/Dialog';

const NUM_SLIDERS = 30;

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

function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(
    () => {
      const handler = setTimeout(() => {
        setDebouncedValue(value);
      }, delay);

      return () => {
        clearTimeout(handler);
      };
    },
    [value, delay]
  );

  return debouncedValue;
}

// Draggable Dialog Title
const DraggableDialogTitle = (props) => {
  const { children, ...rest } = props;

  const handleMouseDown = (event) => {
    // Prevent Dialog from getting focus
    event.preventDefault();
  };

  return (
    <DialogTitle {...rest} style={{ cursor: 'move' }} onMouseDown={handleMouseDown}>
      {children}
    </DialogTitle>
  );
};

const DraggablePaper = (props) => {
  return (
    <Draggable
      handle="#draggable-dialog-title"
      cancel={'[class*="MuiDialogContent-root"]'}
    >
      <Paper {...props} />
    </Draggable>
  );
};

// Multiple Sliders component
const SlidersArray = ({ sliders, setSliders, directions }) => {
  const handleChange = (index, newValue) => {
    sliders[index] = newValue;

    let squaredSliders = sliders.map(a => a * a);
    let squaredSum = squaredSliders.reduce((a, b) => a + b, 0);
    if (squaredSum !== 1) {
      let updatedSliders = sliders.map((value) => {
        return value / Math.sqrt(squaredSum);
      });
      setSliders(updatedSliders);
    } else {
      setSliders(prevState => {
        let newState = [...prevState];
        newState[index] = newValue;
        return newState;
      });
    }
  }

  const getLabel = (index) => {
    return directions[index]?.name || `PCA component ${index}`
  }

  return (
    <Box sx={{ width: 200 }}>
      {sliders.map((sliderValue, index) => (
        <Box key={`slider-${index}`} sx={{ mt: 2 }}>
          <Typography id={`slider-${index}-label`}>
            {`${getLabel(index)}: ${sliderValue.toFixed(2)}`}
          </Typography>
          <Slider
            min={-1}
            max={1}
            step={0.05}
            value={sliderValue}
            onChange={(event, newValue) => handleChange(index, newValue)}
            aria-labelledby={`slider-${index}-label`}
          />
        </Box>
      ))}
    </Box>
  );
};

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

const MemoizedDialogContent = React.memo(({
  directionSliders, 
  setDirectionSliders, 
  allDirections, 
  direction,
  selectedType,
  selectedHead,
  username,
}) => {

  const [dialogDirectionName, setDialogDirectionName] = useState("");
  const [dialogDirectionDescription, setDialogDirectionDescription] = useState("");

  const handleDirectionNameChange = useCallback((event) => {
    setDialogDirectionName(event.target.value);
  }, []);
  
  const handleDirectionDescriptionChange = useCallback((event) => {
    setDialogDirectionDescription(event.target.value);
  }, []);

  const handleSaveDirection = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/api/directions", {
        json: {
          model_name: "gpt2-small",
          type: selectedType,
          head: selectedHead,
          direction: direction.direction,
          username: username,
          direction_name: dialogDirectionName,
          direction_description: dialogDirectionDescription,
        }
      }); 
    } catch (error) {
      console.error(error);
    }
  }


  return (
    <DialogContent dividers={true} style={{height: '300px'}}>
      <Grid container spacing={3} justify="center">
        <Grid item>
          {/* Give your direction a name */}
          <TextField label="Direction name" variant="filled" style={{backgroundColor: '#3a3a3a'}} fullWidth value={dialogDirectionName} onChange={e => setDialogDirectionName(e.target.value)}/>
        </Grid>
        <Grid item>
          {/* Give your direction a description */}
          <TextField label="Direction description" variant="filled" style={{backgroundColor: '#3a3a3a'}} fullWidth multiline value={dialogDirectionDescription} onChange={e => setDialogDirectionDescription(e.target.value)}/>
        </Grid>
        <Grid item>
          {/* Save your direction */}
          <Button variant="outlined" onClick={() => handleSaveDirection('direction name', 'direction description')}>Save Direction</Button>
        </Grid>
        <Grid item>
          <SlidersArray sliders={directionSliders} setSliders={setDirectionSliders} directions={allDirections}/>
        </Grid>
      </Grid>
    </DialogContent>
  )
})

const ColoredResidBox = ({ resid, minDotProduct, maxDotProduct }) => {
  const { dotProduct } = resid;
  const normalizedDotProduct = (dotProduct - minDotProduct) / (maxDotProduct - minDotProduct);
  const colorScale = chroma.scale(['red', 'white', 'blue']).mode('lch');
  const color = colorScale(normalizedDotProduct).hex();
  const textColor = chroma.contrast(color, 'white') > chroma.contrast(color, 'black') ? 'white' : 'black';

  return (
    <Box px={0.1} py={0.2} m={0} bgcolor={color} color={textColor} component={Paper} sx={{borderRadius: 0, border: '1px solid black'}}>
      <Typography variant="body1"><pre style={{fontFamily: 'Times New Roman', margin: 0}}>{resid.decodedToken}</pre></Typography>
    </Box>
  );
};

const PromptRow = ({ promptId, resids, maxDotProduct, minDotProduct }) => (
  <Box display="flex" flexWrap="wrap">
    {resids.map((resid) => <ColoredResidBox key={resid.id} resid={resid} maxDotProduct={maxDotProduct} minDotProduct={minDotProduct} />)}
  </Box>
);

const App = () => {
  const [directionSliderDialogOpen, setDirectionSliderDialogOpen] = useState(false);
  const [selectedType, setSelectedType] = useState("");
  const [selectedHead, setSelectedHead] = useState("");
  const [selectedComponentIndex, setSelectedComponentIndex] = useState("");
  const [resids, setResids] = useState([]);
  const [direction, setDirection] = useState([]);
  const [allDirections, setAllDirections] = useState([]);
  const [maxDotProduct, setMaxDotProduct] = useState(1);
  const [minDotProduct, setMinDotProduct] = useState(-1);
  const [directionSliders, setDirectionSliders] = useState(Array(NUM_SLIDERS).fill(0));

  // Store the user's username in local storage
  const [username, setUsername] = useState(localStorage.getItem('username') || '');
  const debouncedUsername = useDebounce(username, 500);

  const handleUsernameChange = (event) => {
    setUsername(event.target.value);
    // localStorage.setItem('username', event.target.value);
  };

  const handleOpenDirectionSliderDialog = () => {
    let newSliders = Array(NUM_SLIDERS).fill(0);
    newSliders[selectedComponentIndex] = 1;
    setDirectionSliders(newSliders);
    setDirectionSliderDialogOpen(true);
  };

  const handleCloseDirectionSliderDialog = () => {
    setDirectionSliderDialogOpen(false);
  };

  useEffect(
    () => {
      if ((!debouncedUsername) || (debouncedUsername === 'undefined')) return;
      // Now we call our API or do whatever we need with the debounced value
      localStorage.setItem('username', debouncedUsername);
    },
    [debouncedUsername]  // Only call effect if debounced search term changes
  );

  useEffect(
    () => {
      console.log('Triggering!')
      console.log(localStorage.getItem('username'))
      setUsername(localStorage.getItem('username') || '');
    },
    []
  )

  const computeDirection = () => {
    // console.log('direction: ', direction?.direction)
    if (!direction?.direction?.length) return null;
    console.log('All directions: ', allDirections)
    let newDirection = Array(direction?.direction?.length).fill(0);
    for (let i = 0; i < NUM_SLIDERS; i++) {
      for (let j = 0; j < direction?.direction?.length; j++) {
        newDirection[j] += directionSliders[i] * allDirections[i]?.direction?.[j];
      }
    }
    calculateDotProducts(resids, { direction: newDirection });
    return newDirection;
  }

  useEffect(() => {
    const computedDirection = computeDirection()
    // console.log('Computed direction: ', computedDirection)
    if (computedDirection) setDirection({direction: computedDirection});
    // eslint-disable-next-line
  }, [directionSliders]);

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

  const fetchAllDirections = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/api/all_directions", {
        params: {
          model_name: "gpt2-small",
          type: selectedType,
          head: selectedHead,
        },
      });
      console.log(response.data)
      
      setAllDirections(response.data);
    } catch (error) {
      console.error(error);
    }
  }

  const fetchResidsAndDirection = async () => {
    console.log('Fetching resids and direction!')
    try {
      const residResponse = await axios.get("http://127.0.0.1:5000/api/resids", {
        params: {
          model_name: "gpt2-small",
          type: selectedType,
          head: selectedHead,
          component_index: selectedComponentIndex,
        },
      });
      const newResids = residResponse.data;

      const directionResponse = await axios.get("http://127.0.0.1:5000/api/directions", {
        params: {
          model_name: "gpt2-small",
          type: selectedType,
          head: selectedHead,
          component_index: selectedComponentIndex,
        },
      });
      const newDirection = directionResponse.data;
      calculateDotProducts(newResids, newDirection);
      // setResids(newResids);
      setDirection(newDirection);
    } catch (error) {
      console.error(error);
    }
    fetchAllDirections();
  };

  const fetchDirection = async () => {
    console.log('Fetching direction!')    
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

    fetchAllDirections();
  };

  useEffect(() => {
    fetchResidsAndDirection();
    // eslint-disable-next-line
  }, [selectedType, selectedHead]);

  useEffect(() => {
    if (!resids?.length) {
      fetchResidsAndDirection();
    }
    else {
      fetchDirection();
    }
    // eslint-disable-next-line
  }, [selectedComponentIndex]);

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
          <Grid item xs={12} md={6}>
            <TextField value={username} label="Your username" variant="filled" style={{backgroundColor: '#3a3a3a'}} fullWidth onChange={handleUsernameChange}/>
          </Grid>
        </Grid>
        <br />
        {selectedType && (!needsHead || selectedHead) && <Grid container spacing={3} justify="center">
          <Grid item xs={12} md={6}>
            <Button variant="outlined" onClick={handleOpenDirectionSliderDialog}>
              Find a new direction
            </Button>
          </Grid>
        </Grid>}
        <Dialog
          open={directionSliderDialogOpen}
          onClose={handleCloseDirectionSliderDialog}
          PaperComponent={DraggablePaper}
          aria-labelledby="draggable-dialog-title"
        >
          <Draggable handle="#draggable-dialog-title" cancel={'[class*="MuiDialogContent-root"]'}>
            <Paper>
              <DraggableDialogTitle id="draggable-dialog-title">
                Find a new direction
              </DraggableDialogTitle>
              <MemoizedDialogContent 
                directionSliders={directionSliders} 
                setDirectionSliders={setDirectionSliders} 
                allDirections={allDirections}  
                direction={direction}
                selectedType={selectedType}
                selectedHead={selectedHead}
                username={username}
              />
            </Paper>
          </Draggable>
        </Dialog>
        <br />
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
