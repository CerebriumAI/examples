import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import posthog from 'posthog-js';
import App from './App.tsx';
import './index.css';

posthog.init('phc_1J7QtQQ5XMVdWUH940J6qlzuV6PWVLuWZe6RMUs0oLX', { 
  api_host: 'https://us.i.posthog.com',
  person_profiles: 'identified_only'
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
