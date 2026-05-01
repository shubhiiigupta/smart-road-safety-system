import React, { useState, useEffect } from 'react';
import { 
  FiShield, 
  FiSmartphone, 
  FiTruck, 
  FiActivity, 
  FiAlertTriangle 
} from 'react-icons/fi';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';
import './index.css';

function App() {
  const [metrics, setMetrics] = useState({
    total_vehicles: 0,
    helmet_violations: 0,
    distracted_drivers: 0,
    weekly_data: []
  });

  // Fetch live stats from Flask backend
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch('http://localhost:5000/api/stats');
        const data = await res.json();
        setMetrics(data);
      } catch (err) {
        console.error("Failed to fetch stats", err);
      }
    };
    
    // Poll every 2 seconds
    fetchStats();
    const interval = setInterval(fetchStats, 2000);
    return () => clearInterval(interval);
  }, []);

  // Calculate safe percentage
  const totalViolations = metrics.helmet_violations + metrics.distracted_drivers;
  const safePercentage = metrics.total_vehicles === 0 ? 100 : Math.max(0, Math.round(((metrics.total_vehicles - totalViolations) / metrics.total_vehicles) * 100));
  
  // Calculate stroke dashoffset for the SVG ring
  const circleCircumference = 502; // 2 * pi * r (where r=80)
  const strokeDashoffset = circleCircumference - (safePercentage / 100) * circleCircumference;
  return (
    <div className="dashboard-container">
      {/* Header */}
      <header className="dashboard-header">
        <div>
          <h1>
            <FiShield size={48} color="#FFC000" />
            SafetyHero
          </h1>
          <p>Smart Road Safety System Dashboard</p>
        </div>
        <div className="status-badge">
          <div className="dot active"></div>
          System Live
        </div>
      </header>

      {/* Metrics Grid */}
      <div className="metrics-grid">
        <div className="glass-card">
          <div className="card-header">
            <h3>Total Vehicles</h3>
            <div className="icon-wrapper icon-yellow">
              <FiTruck size={24} />
            </div>
          </div>
          <div className="metric-value">{metrics.total_vehicles}</div>
          <div className="metric-label">Vehicles scanned today</div>
        </div>

        <div className="glass-card">
          <div className="card-header">
            <h3>Helmet Violations</h3>
            <div className="icon-wrapper icon-red">
              <FiAlertTriangle size={24} />
            </div>
          </div>
          <div className="metric-value" style={{ color: 'var(--primary-bg)' }}>{metrics.helmet_violations}</div>
          <div className="metric-label">Without helmets</div>
        </div>

        <div className="glass-card">
          <div className="card-header">
            <h3>Distracted Drivers</h3>
            <div className="icon-wrapper icon-yellow">
              <FiSmartphone size={24} />
            </div>
          </div>
          <div className="metric-value">{metrics.distracted_drivers}</div>
          <div className="metric-label">Mobile phone usage</div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="main-content">
        
        {/* Left Column: Live Feed & Analytics */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '25px' }}>
          <div className="glass-card">
            <div className="card-header">
              <h3>Live Camera Feed</h3>
              <div className="icon-wrapper icon-blue">
                <FiActivity size={24} />
              </div>
            </div>
            <div className="live-feed-placeholder">
              {/* Native MJPEG Stream directly from Flask! */}
              <img src="http://localhost:5000/api/video_feed" alt="Live YOLO Camera Feed" />
              <div style={{ position: 'absolute', top: 15, right: 15, background: 'rgba(0,0,0,0.6)', color: 'white', padding: '5px 12px', borderRadius: 20, fontSize: '0.8rem', fontWeight: 'bold' }}>
                REC •
              </div>
            </div>
          </div>

          <div className="glass-card">
            <div className="card-header">
              <h3>Weekly Violations Overview</h3>
            </div>
            <div style={{ width: '100%', height: 250 }}>
              <ResponsiveContainer>
                <BarChart data={metrics.weekly_data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
                  <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: 'var(--text-muted)' }} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fill: 'var(--text-muted)' }} />
                  <Tooltip cursor={{ fill: 'rgba(255, 192, 0, 0.1)' }} contentStyle={{ borderRadius: 12, border: 'none', boxShadow: '0 5px 15px rgba(0,0,0,0.1)' }} />
                  <Bar dataKey="violations" fill="var(--primary-bg)" radius={[6, 6, 6, 6]} barSize={30} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Right Column: Safety Score */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '25px' }}>
          <div className="glass-card">
            <div className="card-header">
              <h3>Today's Safety Score</h3>
            </div>
            <div className="progress-ring-container">
              {/* Simple SVG Circular Progress Ring */}
              <svg width="200" height="200" viewBox="0 0 200 200">
                <circle cx="100" cy="100" r="80" fill="none" stroke="#f0f0f0" strokeWidth="20" />
                <circle 
                  cx="100" cy="100" r="80" 
                  fill="none" 
                  stroke="var(--accent-green)" 
                  strokeWidth="20" 
                  strokeDasharray="502" 
                  strokeDashoffset={strokeDashoffset}
                  strokeLinecap="round"
                  transform="rotate(-90 100 100)"
                />
              </svg>
              <div className="progress-text">
                <h4>{safePercentage}%</h4>
                <p>Safe Riders</p>
              </div>
            </div>
          </div>
          
          <div className="glass-card">
             <div className="card-header">
              <h3>Recent Alerts</h3>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px', marginTop: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px', padding: '10px', background: '#fff0f0', borderRadius: '12px' }}>
                <FiAlertTriangle color="var(--primary-bg)" size={20} />
                <div>
                  <div style={{ fontWeight: 'bold', fontSize: '0.9rem' }}>No Helmet Detected</div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Cam 04 • 2 mins ago</div>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px', padding: '10px', background: '#fff9e6', borderRadius: '12px' }}>
                <FiSmartphone color="var(--accent-yellow)" size={20} />
                <div>
                  <div style={{ fontWeight: 'bold', fontSize: '0.9rem' }}>Mobile Usage Detected</div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Cam 02 • 15 mins ago</div>
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;
