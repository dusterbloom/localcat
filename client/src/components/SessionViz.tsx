import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

// Types for session analytics from f3aff26 (session_store.get_user_session_stats())
interface SessionData {
  total_sessions: number;
  total_time_spent_seconds: number;
  recent_sessions: number;
  current_duration: number;
  turn_count: number;
  analytics: {
    avg_response_time_ms: number;
  };
}

// Mock data - integrate with actual API call to /api/sessions (from session_store)
const mockSessions = [
  { name: 'Session 1', turns: 5, duration: 120 },
  { name: 'Session 2', turns: 8, duration: 180 },
  { name: 'Session 3', turns: 12, duration: 240 },
  { name: 'Session 4', turns: 7, duration: 150 },
  { name: 'Session 5', turns: 15, duration: 300 },
];

const SessionViz: React.FC = () => {
  const [sessionData, setSessionData] = React.useState<SessionData | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    // Fetch from server API (integrate with session_store analytics from f3aff26)
    fetch('/api/sessions/stats')
      .then(res => res.json())
      .then(data => {
        setSessionData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch session data:', err);
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading session analytics...</div>;
  if (!sessionData) return <div>No session data available</div>;

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Session Analytics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div>Total Sessions: {sessionData.total_sessions}</div>
          <div>Total Time: {(sessionData.total_time_spent_seconds / 3600).toFixed(1)} hours</div>
          <div>Recent Sessions: {sessionData.recent_sessions}</div>
          <div>Current Turns: {sessionData.turn_count}</div>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={mockSessions}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis yAxisId="turns" orientation="left" />
            <YAxis yAxisId="duration" orientation="right" />
            <Tooltip />
            <Legend />
            <Line yAxisId="turns" type="monotone" dataKey="turns" stroke="#8884d8" name="Turns" />
            <Line yAxisId="duration" type="monotone" dataKey="duration" stroke="#82ca9d" name="Duration (s)" strokeWidth={2} yAxisId="duration" />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-sm text-muted-foreground mt-2">Avg Response Time: {sessionData.analytics.avg_response_time_ms}ms</p>
      </CardContent>
    </Card>
  );
};

export default SessionViz;
 
