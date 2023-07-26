import React from 'react';
import { Link } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import Box from '@mui/material/Box';
import logo from './logos/cclogo_blue.svg';
import './App.css'

import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Create a dark theme
const navDarkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});

const NavBar = () => {
  return (
    <ThemeProvider theme={navDarkTheme}>
      <CssBaseline />
      <AppBar position="static" color="inherit">
        <Toolbar>
          <Box display="flex" alignItems="center">
            <Typography variant="h3" component="div" sx={{ fontFamily: "Big Shoulders Inline Text", color: 'white' }}>
              Residual
            </Typography>
            <img src={logo} alt="logo" height="50px" />
            <Typography variant="h3" component="div" sx={{ fontFamily: "Big Shoulders Inline Text", color: 'white' }}>
              Viewer
            </Typography>
          </Box>
          <Divider orientation="vertical" flexItem sx={{ mx: 2 }} />
          <Tabs>
            <Tab label="Stream Viewer" component={Link} to="/" />
            <Tab label="Usage Guide" component={Link} to="/usage-guide" />
          </Tabs>
        </Toolbar>
      </AppBar>
    </ThemeProvider>
  );
};

export default NavBar;
