import React, { useState, useEffect, useCallback, useRef } from 'react';
import {BrowserRouter as Router, Route, Routes,} from 'react-router-dom';
import NavBar from './NavBar';
import {
  Button,
  Box,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Grid,
  Paper,
  TextField,
  Slider,
  Card, CardContent
} from '@mui/material';
import axios from 'axios';
import { gpt2_types } from './gpt2_types';
import { Typography } from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import chroma from 'chroma-js';
import Draggable from 'react-draggable';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import Dialog from '@mui/material/Dialog';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { FaTrash } from 'react-icons/fa';
import Popover from '@mui/material/Popover';


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

export async function callApi(method, url, data = null, startMessage = null, endMessage = 'Success!') {
  if (startMessage) {
    toast(startMessage);
  }

  const response = await fetch(url, {
    method,
    body: data ? JSON.stringify(data) : null,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (response.status >= 400 && response.status < 600) {
    const errorData = await response.json();
    toast.error(`An error occurred: ${errorData.error}`);
    throw new Error(`HTTP Error: ${response.status}`);
  }

  if (endMessage) {
    toast(endMessage);
  }

  const result = await response.json();
  return result;
}

function LoadingIndicator() {
  return <h2>Loading...</h2>;
}

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
            disabled={sliderValue === 1 || sliderValue === -1}
          />
        </Box>
      ))}
    </Box>
  );
};

const TypeSelector = ({ types, selectedType, onTypeChange }) => {
const niceTypeName = (badTypeName) => {
  const nameDictionary = {
    'hook_pos_embed': 'Positional Embedding',
    'hook_embed': 'Token Embedding',
  }
  const newName = nameDictionary?.[badTypeName];
  return newName ? newName : badTypeName;
};


  return (
      <Grid item xs={12} md={4}>
        <FormControl variant="filled" fullWidth style={{backgroundColor: '#3a3a3a'}}>
          <InputLabel style={{ color: 'white' }}>Type</InputLabel>
          <Select
            value={selectedType}
            onChange={(event) => onTypeChange(event.target.value)}
          >
            {types.map((type) => <MenuItem value={type} key={type}>
            <Typography style={{ color: 'white' }}>{niceTypeName(type)}</Typography>
            </MenuItem>)}
          </Select>
        </FormControl>
      </Grid>
  );
};

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

const MyDirectionsWidget = ({ direction, setDirection, myDirections, setMyDirections, debouncedUsername }) => {  
  useEffect(() => {
    const fetchData = async () => {
      const result = await axios.get("http://127.0.0.1:5000/api/my_directions", {params: {username: debouncedUsername}});
      setMyDirections(result.data);
    };
    
    fetchData();
  }, [debouncedUsername]); // Empty dependency array means this effect runs once on mount
  
  const handleDelete = async (id) => {
    await callApi('DELETE', `http://127.0.0.1:5000/api/directions/${id}`);
    setMyDirections(myDirections.filter(direction => direction.id !== id));
  };

  console.log(myDirections);
  if (!myDirections) return null;

  return (
    <div>
      <Card>
        <CardContent>
          <Typography variant="body1" color="text.secondary">
            My Directions
          </Typography>
        </CardContent>
        <CardContent>
          <div style={{ overflowY: 'auto' }}>
          {myDirections.map((dir) => (
            <div
              key={dir.id}
              onClick={() => setDirection(dir)}
              style={{
                border: dir?.id === direction?.id ? '1px solid blue' : 'none',
                padding: '10px',
                margin: '10px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}
            >
              <span>{dir.name}</span>
              <FaTrash onClick={() => handleDelete(dir.id)} />
            </div>
          ))}
        </div>
        </CardContent>
      </Card>
    </div>
  );
};


const DirectionInfo = ({ direction }) => {
  if (!direction?.id) return null;
  const directionName = direction?.name || `PCA component ${direction?.componentIndex}`;
  const percentVariance = direction?.fractionOfVarianceExplained ? Math.round(direction.fractionOfVarianceExplained * 10000) / 100 : null;
  console.log(direction?.myDescription);
  console.log(direction);
  return (
    <>
      <Grid container direction="column" spacing={2}>
        <Grid item xs={12}>
          <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
            {`Direction: ${directionName}`}
          </Typography>
        </Grid>
        {percentVariance !== null ? <Grid item xs={12}>
          <Typography variant="body1">
            {`This direction explains ${percentVariance} percent of the variance`}
          </Typography>
        </Grid> : null}
        <Grid item xs={24}>
          <Grid container direction="row" spacing={3}>
            {direction?.myDescription ? <Grid item xs={12}>
              <Typography variant="body1">
                {`Your description: ${direction.myDescription.description}`}
              </Typography>
              <VoteWidget descriptionId={direction.myDescription.id} initialVotes={direction.myDescription.upvotes} />
            </Grid> : null}
            {direction?.bestDescription ? <Grid item xs={12}>
              <Typography variant="body1">
                {`Highest-rated description: ${direction.bestDescription.description}`}
              </Typography>
              <VoteWidget descriptionId={direction.bestDescription.id} initialVotes={direction.bestDescription.upvotes} />
            </Grid> : null}
          </Grid>
        </Grid>
      </Grid>
      <br/>
    </>
    );
  }

  function VoteWidget({ descriptionId, initialVotes }) {
    const [votes, setVotes] = useState(initialVotes);
    const [userVote, setUserVote] = useState(null);
  
    useEffect(() => {
      const storedVote = localStorage.getItem(`votes-${descriptionId}`);
      if (storedVote) {
        setUserVote(storedVote);
      }
    }, [descriptionId]);
  
    const handleVote = async (type) => {
      try {
        if (userVote && userVote === type) return;
        if (userVote && userVote !== type) {
          setUserVote(null);
          localStorage.removeItem(`votes-${descriptionId}`);  // Remove vote from local storage.
        }
    
        await callApi('POST', `http://127.0.0.1:5000/api/descriptions/${descriptionId}/${type}`);
    
        if (type === 'upvote') {
          setVotes(prevVotes => prevVotes + 1);
        } else if (type === 'downvote') {
          setVotes(prevVotes => prevVotes - 1);
        }
    
        // Store in local storage.
        if (!userVote) {
          localStorage.setItem(`votes-${descriptionId}`, type);
          setUserVote(type);
        }
      } catch (error) {
        console.error("Error voting: ", error);
      }
    };
    const widgetStyle = {
      display: 'flex',
      flexDirection: 'row',
      alignItems: 'center',
      border: '1px solid gray',
      borderRadius: '5px',
      padding: '10px',
      width: '100px',
      justifyContent: 'space-between'
    };
  
    const voteStyle = (voteType) => ({
      color: userVote === voteType ? (voteType === 'upvote' ? 'green' : 'red') : 'black',
      cursor: 'pointer',
      flex: 1,
      textAlign: 'center'
    });

    const backgroundStyle = (voteType) => ({
      backgroundColor: userVote === voteType ? (voteType === 'upvote' ? '#e6ffe6' : '#ffe6e6') : 'transparent',
      borderRadius: '5px',
      padding: '5px'
    });
  
    return (
      <div style={widgetStyle}>
        <Typography
          onClick={() => handleVote('upvote')}
          style={{ ...voteStyle('upvote'), ...backgroundStyle('upvote') }}
        >
          ⬆️
        </Typography>
        <span style={{ flex: 1, textAlign: 'center' }}>{votes}</span>
        <Typography 
          onClick={() => handleVote('downvote')} 
          style={{ ...voteStyle('downvote'), ...backgroundStyle('downvote') }}
        >
          ⬇️
        </Typography>
      </div>
    );
  }

const MemoizedDialogContent = React.memo(({
  directionSliders, 
  setDirectionSliders, 
  allDirections, 
  direction,
  selectedType,
  selectedHead,
  username,
  setDirection,
  myDirections,
  setMyDirections
}) => {

  const [dialogDirectionName, setDialogDirectionName] = useState("");
  const [dialogDirectionDescription, setDialogDirectionDescription] = useState("");
  const [saving, setSaving] = useState(false);

  const handleDirectionNameChange = useCallback((event) => {
    setDialogDirectionName(event.target.value);
  }, []);
  
  const handleDirectionDescriptionChange = useCallback((event) => {
    setDialogDirectionDescription(event.target.value);
  }, []);

  const handleSaveDirection = async () => {
    try {
      setSaving(true);
      const response = await callApi("POST", "http://127.0.0.1:5000/api/directions", {
        model_name: "gpt2-small",
        type: selectedType,
        head: selectedHead,
        direction: direction.direction,
        username: username,
        direction_name: dialogDirectionName,
        direction_description: dialogDirectionDescription,
      })
      setSaving(false);
      console.log(response);
      console.log(myDirections);
      setDirection(response);
      setMyDirections([...myDirections, response]);
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
          <Button variant="outlined" onClick={() => handleSaveDirection('direction name', 'direction description')}>{'Save Direction'}</Button>
        </Grid>
        <Grid item>
          <SlidersArray sliders={directionSliders} setSliders={setDirectionSliders} directions={allDirections}/>
        </Grid>
      </Grid>
    </DialogContent>
  )
})

const ColoredResidBox = ({ resid, minDotProduct, maxDotProduct}) => {
  const { dotProduct } = resid;
  const normalizedDotProduct = (dotProduct - minDotProduct) / (maxDotProduct - minDotProduct);
  const colorScale = chroma.scale(['red', 'white', 'blue']).mode('lch');
  const color = colorScale(normalizedDotProduct).hex();
  const textColor = chroma.contrast(color, 'white') > chroma.contrast(color, 'black') ? 'white' : 'black';

  const predictedNextTokens = resid.predictedNextTokens;

  const [anchorEl, setAnchorEl] = React.useState(null);
  const [isOverPopover, setIsOverPopover] = React.useState(false);
  const open = Boolean(anchorEl) && predictedNextTokens && Object.keys(predictedNextTokens).length > 0;
  const id = open ? 'simple-popover' : undefined;

  const handleMouseEnterBox = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMouseLeaveBox = () => {
    if (!isOverPopover) {
      setAnchorEl(null);
    }
  };

  const handleMouseEnterPopover = () => {
    setIsOverPopover(true);
  };

  const handleMouseLeavePopover = () => {
    setIsOverPopover(false);
    setAnchorEl(null);
  };

  // Convert the predictedNextTokens object to an array and sort it by value
  const sortedTokens = predictedNextTokens
    ? Object.entries(predictedNextTokens).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <>
      <Box
        px={0.1}
        py={0.2}
        m={0}
        bgcolor={color}
        color={textColor}
        component={Paper}
        sx={{borderRadius: 0, border: '1px solid black'}}
        onMouseEnter={handleMouseEnterBox}
        onMouseLeave={handleMouseLeaveBox}
      >
        <Typography variant="body1"><pre style={{fontFamily: 'Times New Roman', margin: 0}}>{resid.decodedToken}</pre></Typography>
      </Box>
      {/* <Popover
        id={id}
        open={open}
        anchorEl={anchorEl}
        onClose={handleMouseLeavePopover}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'left',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'left',
        }}
        onMouseEnter={handleMouseEnterPopover}
        onMouseLeave={handleMouseLeavePopover}
      >
        <Box p={2}>
          <Typography variant="h6">Predicted token after "{resid.decodedToken}":</Typography>
          {sortedTokens.map(([token, probability]) => (
            <Typography key={token}>"{token}": {probability}</Typography>
          ))}
        </Box>
      </Popover> */}
    </>
  );
};



function PromptRow({ promptId, resids, maxDotProduct, minDotProduct }) {
  const [currentHoveredResid, setCurrentHoveredResid] = React.useState(null);
  
  return (
    <Box display="flex" flexWrap="wrap">
      {resids.map((resid) => 
        <ColoredResidBox 
          key={resid.id} 
          resid={resid} 
          maxDotProduct={maxDotProduct} 
          minDotProduct={minDotProduct} 
          />)}
    </Box>
  );
}

function PromptTable({ groupedResids, maxDotProduct, minDotProduct }) {
  return (
    <Grid container spacing={1}>
      {Object.entries(groupedResids).reverse().map(([promptId, resids]) => (
        <Grid item xs={12} key={promptId}>
          <PromptRow 
            promptId={promptId} 
            resids={resids} 
            maxDotProduct={maxDotProduct} 
            minDotProduct={minDotProduct} 
          />
        </Grid>
      ))}
    </Grid>
  );
};

const DirectionDescriptionField = ({direction, setDirection, username}) => {
  const [description, setDescription] = useState(direction?.description || "");

  const handleDescriptionChange = useCallback((event) => {
    setDescription(event.target.value);
  }, []);

  const handleSaveDescription = async () => {
    try {
      await callApi('POST', `http://127.0.0.1:5000/api/directions/${direction?.id}/descriptions`, {
        direction_description: description,
        username: username,
      }).then((direction) => {
        setDescription('');
        setDirection({...direction});
      });
    } catch (error) {
      console.error(error);
    }
  }

  if (!direction?.id) return null;

  return (
    <>
      <Grid container direction="column" spacing={1}>
        <Grid item>
          <TextField 
            label="Describe this direction!" 
            variant="outlined" 
            size="medium"
            InputProps={{
              style: {fontSize: "14px"}
            }}
            InputLabelProps={{
              style: {fontSize: "14px"}
            }}
            style={{
              borderRadius: '5px'
            }} 
            fullWidth 
            multiline 
            rows={6} // increase the number of rows to increase height
            value={description} 
            onChange={handleDescriptionChange}
          />
        </Grid>
        <Grid item>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={handleSaveDescription}
          >
            Submit
          </Button>
        </Grid>
      </Grid>
    </>
  )
}

const AddYourOwnPromptField = ({username, fetchResidsAndDirection}) => {
  const [prompt, setPrompt] = useState("");

  const handlePromptChange = useCallback((event) => {
    setPrompt(event.target.value);
  }, []);

  const handleSavePrompt = async () => {
    try {
      await callApi('POST', `http://127.0.0.1:5000/api/prompts`, {
        prompt,
        username,
        model_name: 'gpt2-small'
      });
    } catch (error) {
      console.error(error);
    }

    fetchResidsAndDirection();
  }

  return (
    <>
      <Grid container direction="column" spacing={1}>
        <Grid item>
          <TextField 
            label="Submit your own prompt!" 
            variant="outlined" 
            size="medium"
            InputProps={{
              style: {fontSize: "14px"}
            }}
            InputLabelProps={{
              style: {fontSize: "14px"}
            }}
            style={{
              borderRadius: '5px'
            }} 
            fullWidth 
            multiline 
            rows={6} // increase the number of rows to increase height
            value={prompt} 
            onChange={handlePromptChange}
          />
        </Grid>
        <Grid item>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={handleSavePrompt}
          >
            Submit
          </Button>
        </Grid>
      </Grid>
    </>
  )
}

const App = () => {
  return (
    <Router>
      <NavBar />
      <Routes>
        <Route path="/" element={<MainStreamViewerPage />} />
        <Route path="/usage-guide" element={<UsageGuidePage />} />
      </Routes>
    </Router>
  );
};

const UsageGuidePage = () => {
  return(
      <>
        <Card>
          <CardContent>
            Todo
          </CardContent>
        </Card>
      </>
  );
}


const MainStreamViewerPage = () => {
  const [directionSliderDialogOpen, setDirectionSliderDialogOpen] = useState(false);
  const [selectedType, setSelectedType] = useState("ln_final.hook_normalized");
  const [selectedHead, setSelectedHead] = useState("");
  const [selectedComponentIndex, setSelectedComponentIndex] = useState("0");
  const [resids, setResids] = useState([]);
  const [direction, setDirection] = useState([]);
  const [allDirections, setAllDirections] = useState([]);
  const [maxDotProduct, setMaxDotProduct] = useState(1);
  const [minDotProduct, setMinDotProduct] = useState(-1);
  const [directionSliders, setDirectionSliders] = useState(Array(NUM_SLIDERS).fill(0));
  const [loadingResids, setLoadingResids] = useState(false);
  const [myDirections, setMyDirections] = useState([]);

  const [hasFetchedOnce, setHasFetchedOnce] = useState(false);

  const firstRender = useRef(true);

  const typeShape = gpt2_types[selectedType];

  const needsHead = !!typeShape?.includes(12);

  console.log('Resids:', resids);
  console.log('Direction:', direction);

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

  useEffect(() => {
    calculateDotProducts(resids, direction);
  }, [direction?.id])

  const calculateDotProducts = (residsToCalculate, directionToCalculate) => {
    if (!residsToCalculate?.length || !directionToCalculate?.direction) return;
    console.log('calculating dot products')
    const dotProducts = residsToCalculate.map(resid => {
      let dotProduct = 0;
      if (resid.resid.length === directionToCalculate.direction.length) {
        dotProduct = resid.resid.reduce((sum, value, i) => sum + value * directionToCalculate.direction[i], 0);
      }
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
          head: needsHead ? selectedHead : null,
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
    if (!selectedType || [null, undefined, ""].includes(selectedComponentIndex)) return;
    try {
      setLoadingResids(true);
      const residResponse = await axios.get("http://127.0.0.1:5000/api/resids", {
        params: {
          model_name: "gpt2-small",
          type: selectedType,
          head: selectedHead,
          component_index: selectedComponentIndex,
          username: username,
        },
      });
      console.log('Got resid response:', residResponse)
      const newResids = residResponse.data;

      const directionResponse = await axios.get("http://127.0.0.1:5000/api/directions", {
        params: {
          model_name: "gpt2-small",
          type: selectedType,
          head: needsHead ? selectedHead : null,
          component_index: selectedComponentIndex,
        },
      });
      console.log('Got direction response:', directionResponse)
      const newDirection = directionResponse.data;
      calculateDotProducts(newResids, newDirection);
      setLoadingResids(false);
      // setResids(newResids);
      setDirection(newDirection);
    } catch (error) {
      console.error(error);
      setLoadingResids(false);
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
          head: needsHead ? selectedHead : null,
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
    if (!hasFetchedOnce) return;
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }

    fetchResidsAndDirection();
    // eslint-disable-next-line
  }, [selectedType, selectedHead]);

  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }

    if (!resids?.length) {
      fetchResidsAndDirection();
    }
    else {
      fetchDirection();
    }
    setHasFetchedOnce(true);
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
  const maxComponentIndex = (
    typeShape?.includes(64) ? 64 : 
    typeShape?.includes(768) ? 768 :
    typeShape?.includes(3072) ? 3072 :
    null
  )

  const findNewDirection = (
      <>
            {selectedType && (!needsHead || (selectedHead === 0) || selectedHead) && !!resids?.length && <Grid container spacing={3} justify="center">
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
                username={debouncedUsername}
                setDirection={setDirection}
                myDirections={myDirections}
                setMyDirections={setMyDirections}
              />
            </Paper>
          </Draggable>
        </Dialog>
  </>);

  const selectTypeHeadComp = (
      <>
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
        </>
  );

  return (
    <ThemeProvider theme={darkTheme}>
      <div style={appStyle}>
        <ToastContainer />
        <Grid container spacing={1} justify="center" direction={'row'}>
        {/*First Col*/}
        <Grid item container xs={8} direction={'column'}>
          <Grid item>{selectTypeHeadComp}</Grid>
          <Grid item>
            {loadingResids && <>
              <LoadingIndicator />
              <br />
            </>}
          </Grid>
          <Grid item>
            <DirectionInfo direction={direction}/>
          </Grid>
          <Grid item>
            {!loadingResids && <PromptTable
                groupedResids={groupedResids}
                minDotProduct={minDotProduct}
                maxDotProduct={maxDotProduct}
            />}
          </Grid>
        </Grid>
        {/*Second col*/}
        <Grid item container xs={4} direction={'column'} spacing={1}>
        <Grid item>
            {findNewDirection}
          </Grid>
          <Grid item>
            <TextField value={username}
                       label="Your username"
                       variant="filled"
                       style={{backgroundColor: '#3a3a3a'}}
                       fullWidth
                       onChange={handleUsernameChange}
            />
          </Grid>
          <Grid item>
            <DirectionDescriptionField direction={direction} setDirection={setDirection} username={debouncedUsername}/>
          </Grid>
          <Grid item>
            <AddYourOwnPromptField username={debouncedUsername} fetchResidsAndDirection={fetchResidsAndDirection} />
          </Grid>          
            <Grid item>
              <MyDirectionsWidget
                direction={direction}
                setDirection={setDirection}
                myDirections={myDirections}
                setMyDirections={setMyDirections}
                debouncedUsername={debouncedUsername}
              />
          </Grid>
        </Grid>
      </Grid>
      </div>
    </ThemeProvider>
  );
};

export default App;
