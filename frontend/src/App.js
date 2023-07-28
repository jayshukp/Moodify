import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Link, Redirect } from 'react-router-dom';
import './App.css';
import Statistics from './Statistics';

const App = () => {
  const [email, setEmail] = useState('');
  const [loggedIn, setLoggedIn] = useState(false);
  const [artist, setArtist] = useState('');
  const [song, setSong] = useState('');
  const [mood, setMood] = useState('');
  const [explanation, setExplanation] = useState('');
  const [statistics, setStatistics] = useState('');

  useEffect(() => {
    // Check if the user is already logged in (persist login state)
    const storedEmail = localStorage.getItem('email');
    if (storedEmail) {
      setEmail(storedEmail);
      setLoggedIn(true);
    }
  }, []);

  const handleLogin = (e) => {
    e.preventDefault();

    // Perform login authentication
    // Example: Check email against a user database
    // Simulating a successful login for demonstration purposes
    if (email.trim() !== '') {
      setLoggedIn(true);
      localStorage.setItem('email', email);

      // Clear the submission box, mood, explanation, and statistics when logging in with a new email
      setArtist('');
      setSong('');
      setMood('');
      setExplanation('');
      setStatistics('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch('http://192.168.86.26:5001/api/detectMood', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ artist, song }),
      });

      if (response.ok) {
        const data = await response.json();
        setMood(data.mood);
        setExplanation(data.explanation);
        setStatistics(data.statistics);
      } else {
        throw new Error('Request failed');
      }
    } catch (error) {
      console.log(error);
    }
  };

  const handleLogout = () => {
    // Perform logout action
    setLoggedIn(false);
    setEmail('');
    localStorage.removeItem('email');

    // Clear the submission box, mood, explanation, and statistics when logging out
    setArtist('');
    setSong('');
    setMood('');
    setExplanation('');
    setStatistics('');
  };

  return (
    <Router>
      <div className='app'>
        <header className='header'>
          <h1 className='title'>
            <Link to='/' className='title-link'>
              Song Mood Detector
            </Link>
          </h1>
          {loggedIn && (
            <div className='user-info'>
              <p className='user-email'>{email}</p>
              <button className='logout-button' onClick={handleLogout}>
                Logout
              </button>
            </div>
          )}
        </header>

        <Route exact path='/'>
          {loggedIn ? (
            <Redirect to='/submit' />
          ) : (
            <main className='main'>
              <form className='form' onSubmit={handleLogin}>
                <div className='input-group'>
                  <label className='input-label'>
                    Email:
                    <input
                      className='input-field'
                      type='email'
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                    />
                  </label>
                </div>

                <button className='login-button' type='submit'>
                  Enter
                </button>
              </form>
            </main>
          )}
        </Route>
        <Route path='/submit'>
          {loggedIn ? (
            <main className='main'>
              <form className='form' onSubmit={handleSubmit}>
                <div className='input-group'>
                  <label className='input-label'>
                    Artist:
                    <input
                      className='input-field'
                      type='text'
                      value={artist}
                      onChange={(e) => setArtist(e.target.value)}
                      required
                    />
                  </label>
                </div>

                <div className='input-group'>
                  <label className='input-label'>
                    Song:
                    <input
                      className='input-field'
                      type='text'
                      value={song}
                      onChange={(e) => setSong(e.target.value)}
                      required
                    />
                  </label>
                </div>

                <button className='detect-button' type='submit'>
                  Detect Mood
                </button>
              </form>

              {mood && (
                <div className='result'>
                  <h2 className='result-heading'>Mood: {mood}</h2>
                  <p className='result-text'>Explanation: {explanation}</p>
                  <p className='result-text'>Stats: {statistics}</p>
                </div>
              )}
            </main>
          ) : (
            <Redirect to='/' />
          )}
        </Route>
        <Route path='/statistics'>
          {loggedIn ? (
            <Statistics statistics={statistics} />
          ) : (
            <Redirect to='/' />
          )}
        </Route>

        <footer className='footer'>
          <p className='footer-text'>Designed by Jayshuk Pandrangi</p>
        </footer>
      </div>
    </Router>
  );
};

export default App;
