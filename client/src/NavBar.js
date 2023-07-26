import React from 'react';
import { Link } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';

const NavBar = () => {
  return (
    <AppBar position="static">
      <Tabs>
        <Tab label="Main Stream Viewer" component={Link} to="/" />
        <Tab label="Usage Guide" component={Link} to="/usage-guide" />
      </Tabs>
    </AppBar>
  );
};

export default NavBar;
