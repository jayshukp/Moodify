// SongStatistics.js

import React, { useState } from 'react';
import StatisticsDashboard from './StatisticsDashboard';

const SongStatistics = () => {
  const [audioFeaturesData, setAudioFeaturesData] = useState([]);

  const handleSongSubmit = (event) => {
    event.preventDefault();
    const songName = event.target.song.value;

    // Fetch the audio features data for the given song from your backend
    fetch(`/api/getAudioFeatures?song=${songName}`)
      .then((response) => response.json())
      .then((data) => setAudioFeaturesData(data));
  };

  return (
    <div>
      <form onSubmit={handleSongSubmit}>
        <input type="text" name="song" placeholder="Enter song name" />
        <button type="submit">Submit</button>
      </form>

      <StatisticsDashboard audioFeaturesData={audioFeaturesData} />
    </div>
  );
};

export default SongStatistics;
