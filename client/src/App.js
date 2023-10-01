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
import { API_URL } from './settings';
import Switch from '@mui/material/Switch';
import { Link } from 'react-router-dom';


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

const TypeSelector = ({ types, selectedType, onTypeChange, advancedMode }) => {
  const niceTypeName = (badTypeName) => {
    const nameDictionary = {
      'hook_pos_embed': 'Positional Embedding',
      'hook_embed': 'Token Embedding',
    }
    const newName = nameDictionary?.[badTypeName];
    return newName ? newName : badTypeName;
  };

  const normalModeNameDictionary = {
    'hook_pos_embed': 'Positional Embedding',
    'hook_embed': 'Token Embedding',
    'blocks.0.hook_resid_pre': 'Layer 0',
    'blocks.1.hook_resid_pre': 'Layer 1',
    'blocks.2.hook_resid_pre': 'Layer 2',
    'blocks.3.hook_resid_pre': 'Layer 3',
    'blocks.4.hook_resid_pre': 'Layer 4',
    'blocks.5.hook_resid_pre': 'Layer 5',
    'blocks.6.hook_resid_pre': 'Layer 6',
    'blocks.7.hook_resid_pre': 'Layer 7',
    'blocks.8.hook_resid_pre': 'Layer 8',
    'blocks.9.hook_resid_pre': 'Layer 9',
    'blocks.10.hook_resid_pre': 'Layer 10',
    'blocks.11.hook_resid_pre': 'Layer 11',
    'ln_final.hook_normalized': 'Output',
  }

  return (
      <Grid item xs={12} md={4}>
        <FormControl variant="filled" fullWidth style={{backgroundColor: '#3a3a3a'}}>
          <InputLabel style={{ color: 'white' }}>Type</InputLabel>
          <Select
            value={selectedType}
            onChange={(event) => onTypeChange(event.target.value)}
          >
            {advancedMode ? types.map((type) => <MenuItem value={type} key={type}>
            <Typography style={{ color: 'white' }}>{niceTypeName(type)}</Typography>
            </MenuItem>) : types.map((type) => normalModeNameDictionary[type] ? <MenuItem value={type} key={type}>
                <Typography style={{ color: 'white' }}>{normalModeNameDictionary[type]}</Typography>
              </MenuItem> : null)}
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
      const result = await axios.get(`${API_URL}/api/my_directions`, {params: {username: debouncedUsername}});
      setMyDirections(result.data);
    };
    
    fetchData();
  }, [debouncedUsername]); // Empty dependency array means this effect runs once on mount
  
  const handleDelete = async (id) => {
    await callApi('DELETE', `${API_URL}/api/directions/${id}`);
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
    
        await callApi('POST', `${API_URL}/api/descriptions/${descriptionId}/${type}`);
    
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
      const response = await callApi("POST", `${API_URL}/api/directions`, {
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

function UploadDirectionButton({ onUpload }) {
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const parsedData = JSON.parse(e.target.result);
          if (Array.isArray(parsedData.direction) && parsedData.direction.length === 768) { // Adjust 768 if needed
            onUpload(parsedData);
          } else {
            alert('Invalid direction format.');
          }
        } catch (error) {
          alert('Error reading file. Make sure it is a valid JSON.');
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    <div>
      <input 
        type="file" 
        ref={fileInputRef} 
        style={{ display: 'none' }} 
        accept=".json"
        onChange={handleFileChange}
      />
      <Button variant="outlined" onClick={() => fileInputRef.current.click()}>Upload Direction</Button>
    </div>
  );
}

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
      await callApi('POST', `${API_URL}/api/directions/${direction?.id}/descriptions`, {
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
      await callApi('POST', `${API_URL}/api/prompts`, {
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
  const darkTheme = {
    backgroundColor: "#333", // dark gray background
    color: "#FFF", // white text
    padding: '20px'
  };
  
  return (
      <>
        <Card style={darkTheme}>
          <CardContent>
            <Typography paragraph>
              A YouTube tutorial of the Residual Stream viewer is available <Link external to="https://www.youtube.com/watch?v=9d2fs7kt1I0&ab_channel=AdamYedidia">here</Link>.
            </Typography>
            <Typography paragraph>
              The residual stream viewer is a tool for finding interesting directions in the residual stream of GPT2-small,
              {' '}for writing explanations for those directions and reading the explanations left by others,
              {' '}and for constructing new directions out of linear combinations of old ones.
            </Typography>
            <Typography paragraph>
              A more detailed explanation of how transformers networks work and what the residual stream is can be found
              {' '}<Link external to="https://transformer-circuits.pub/2021/framework/index.html">here</Link>. If you want to actually understand what the residual stream is and how transformers work, 
              {' '}the text that follows here is hopelessly insufficient, and you should really follow the earlier link. 
              {' '}However, as 
              {' '}a <emph>very</emph> brief summary of what the "residual stream" is: 
            </Typography> 
            <Typography paragraph>
              The residual stream can be thought of as the
              {' '}intermediate state of the transformer network's computation. It is the output of each layer of the network
              {' '}before it is fed into the next layer. Each prompt is split into "tokens,"
              {' '}i.e. subparts of the prompt that roughly correspond to words or parts of words. At each layer, each token has 
              {' '}its own associated residual stream vector.
              {' '}The residual stream at the beginning of the network, before any layer has acted, is equal to the "Token Embedding",
              {' '}i.e. the "meaning" of that token as encoded by a 768-dimensional vector, plus the "Positional embedding",
              {' '}i.e. the "meaning" of that token's <emph>position in the prompt</emph> as encoded by a 768-dimensional vector.
              {' '}Each layer acts on the residual stream by reading certain parts of the residual stream, doing some computation on them,
              {' '}and then adding the result back into the residual stream. At the end of the network, the residual stream is transformed 
              {' '}into a probability distribution over which token comes next.
            </Typography>
            <Typography paragraph>
              It's not easy to directly interpret a 768-dimensional vector, let alone one at each layer and at each token in the prompt.
              {' '}It's the purpose of this tool to make the job of interpreting such vectors easier. One way of interpreting the residual
              {' '}stream is by considering different possible <emph>directions</emph> in the residual stream. By analogy, imagine if there was
              {' '}a arrow in front of you, oriented somehow in space. The arrow represents the residual stream, by analogy. One way you might approach
              {' '}describing the arrow's direction is by considering how "northerly" the arrow's direction is; that is, to what degree the arrow is pointing
              {' '}North. If the arrow was pointing northward, we might say that the arrow had positive northerliness, and if the arrow was pointing
              {' '}southward, we might say that the arrow had negative northerliness. An arrow pointing northeast could still be said to have positive northerliness;
              {' '}it wouldn't have to be pointing exactly north. If we wanted to classify arrows by their northerliness, and color them accordingly,
              {' '}we might color arrows pointing northwest or northeast or directly north blue, and color arrows pointing southwest or southeast or directly south red.
              {' '}Arrows that pointed in a direction orthogonal to the north-south direction could be left uncolored.
            </Typography>
            <Typography paragraph>
              We can apply the same concept to directions in the residual stream. Unlike an arrow in three-dimensional space, which has three dimensions,
              {' '}the residual stream has 768 dimensions, but the same principle applies. When you choose a direction with the residual stream viewer, 
              {' '}each of residual streams at each token will light up blue or red depending on whether the residual stream vector at that token is pointing a similar
              {' '}direction to that direction, or equivalently, depending on the <Link external to="https://en.wikipedia.org/wiki/Dot_product">dot product</Link> between 
              {' '}the residual stream vector and the direction vector.
            </Typography>
            <Typography paragraph>
              By observing which tokens light up blue and which tokens light up red, you can get a sense of what the direction is doing. For example, a direction that lit up early
              {' '}in the prompt in red and later tokens in blue would probably relate primarily to the positions of the tokens rather than their meanings. Finding interesting and 
              {' '}interpretable directions is hopefully a good way to make interpretability progress.
            </Typography>
            <Typography paragraph>
              How can we find interesting directions? One simple way of finding them is by running <Link external to="https://en.wikipedia.org/wiki/Principal_component_analysis">Principal Components Analysis</Link>, or PCA, on 
              {' '}the residual stream vectors for a given layer. Basically, doing this automatically finds the most interesting directions for us, ranked in decreasing order of how interesting they are.
              {' '}You can look at directions like these using the "Component Index" dropdown. As a concrete example, if you look at Layer 0, Component Index 7, you'll be looking at residual stream vectors from the first layer,
              {' '}and you'll be looking at the eighth-most important "direction", as found by PCA. It's hard to know for sure, but it looks like that direction has at one of its ends auxiliary verbs like "is", "has", or "should",
              {' '}and has proper nouns at the other of its ends.
            </Typography>
            <Typography paragraph>
              If you wanted to combine two or more directions, you could use the "Find a new direction" button. This will open a dialog box where you can find arbitrary linear combinations of PCA directions using the sliders.
              {' '}You'll be able to see the residual stream vectors update in real-time as you move the sliders. You can save those sliders and give them names and descriptions, and upvote descriptions for directions you like.
            </Typography>
            <Typography paragraph>
              Any direction you save or description you give will be associated with your username. This website only requires a username--no password. This means that any other user can see what directions you've saved,
              {' '}simply by typing your username into the username field. Hopefully, that's a feature and not a bug, at least for now! Once you've saved a direction, you'll be able to view it by clicking it on the right side-bar.
            </Typography>
            <Typography paragraph>
              You can also submit your own prompts. If you're curious about your theory of what a direction is doing, and want to test it, you could try submitting your own prompt and seeing if the pattern you've observed fits the prompt you've submitted.
            </Typography>
          </CardContent>
        </Card>
      </>
  );
}


const MainStreamViewerPage = () => {
  const [directionSliderDialogOpen, setDirectionSliderDialogOpen] = useState(false);
  const [selectedType, setSelectedType] = useState("blocks.0.hook_resid_pre");
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

  const [advancedMode, setAdvancedMode] = useState(false);

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

  const generateJSONBlob = (data) => {
    console.log(data);
    const jsonString = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonString], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    return url;
  };

  const downloadDirectionAsJSON = () => {
    const blobURL = generateJSONBlob(direction.direction);
    const tempLink = document.createElement('a');
    tempLink.href = blobURL;
    tempLink.setAttribute('download', 'direction.json');
    tempLink.click();
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
      const response = await axios.get(`${API_URL}/api/all_directions`, {
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
      const residResponse = await axios.get(`${API_URL}/api/resids`, {
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

      const directionResponse = await axios.get(`${API_URL}/api/directions`, {
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
      const response = await axios.get(`${API_URL}/api/directions`, {
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
    console.log(firstRender)
    console.log(hasFetchedOnce)
    if (!hasFetchedOnce) return;
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }

    fetchResidsAndDirection();
    // eslint-disable-next-line
  }, [selectedType, selectedHead]);

  useEffect(() => {
    fetchResidsAndDirection();
    setHasFetchedOnce(true);
  }, [])

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

  const toggleAdvancedMode = () => {
    setAdvancedMode(!advancedMode);
  }

  const findNewDirection = (
      <>
            {selectedType && (!needsHead || (selectedHead === 0) || selectedHead) && !!resids?.length && 
          <Grid item xs={12} md={6}>
            <Button variant="outlined" onClick={handleOpenDirectionSliderDialog}>
              Find a new direction
            </Button>
          </Grid>
        }
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

  const selectTypeHeadComp = (advancedMode) => (
      <>
        <TypeSelector
            advancedMode={advancedMode}
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
          <Grid item>
            <div>
              <span>{advancedMode ? "Advanced Mode" : "Normal Mode" }</span>
              <Switch checked={advancedMode} onChange={toggleAdvancedMode} />
            </div>
          </Grid>
          <Grid item>{selectTypeHeadComp(advancedMode)}</Grid>
          <Grid item>
            {loadingResids && <>
              <LoadingIndicator />
              <br />
            </>}
          </Grid>
          <Grid item>
            <br />
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
        <Grid item container direction="row" spacing={1}>
              {findNewDirection}
            <Grid item xs={12} md={6}>
              <Button variant="outlined" color="primary" onClick={downloadDirectionAsJSON}>
                Download Direction
              </Button>
            </Grid>
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
