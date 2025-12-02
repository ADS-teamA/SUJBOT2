/**
 * Admin Dashboard
 *
 * Shows system statistics:
 * - Total users
 * - Active users
 * - Admin users
 * - Total conversations
 * - Users active in last 24h
 */

import { useEffect, useState } from 'react';
import { Card, CardContent, Typography, Box, Grid, CircularProgress } from '@mui/material';
import { Users, UserCheck, Shield, MessageSquare, Activity } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL !== undefined
  ? import.meta.env.VITE_API_BASE_URL
  : 'http://localhost:8000';

interface Stats {
  total_users: number;
  active_users: number;
  admin_users: number;
  total_conversations: number;
  total_messages: number;
  users_last_24h: number;
  timestamp: string;
}

interface StatCardProps {
  title: string;
  value: number;
  icon: React.ReactNode;
  color: string;
}

const StatCard = ({ title, value, icon, color }: StatCardProps) => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Box display="flex" alignItems="center" justifyContent="space-between">
        <Box>
          <Typography color="textSecondary" gutterBottom variant="body2">
            {title}
          </Typography>
          <Typography variant="h4" component="div">
            {value.toLocaleString()}
          </Typography>
        </Box>
        <Box
          sx={{
            backgroundColor: color,
            borderRadius: '50%',
            p: 1.5,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {icon}
        </Box>
      </Box>
    </CardContent>
  </Card>
);

export const Dashboard = () => {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/admin/stats`, {
          credentials: 'include',
        });

        if (!response.ok) {
          throw new Error('Failed to fetch stats');
        }

        const data = await response.json();
        setStats(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={3}>
        <Typography color="error">Error: {error}</Typography>
      </Box>
    );
  }

  if (!stats) {
    return null;
  }

  return (
    <Box p={3}>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      <Typography variant="body2" color="textSecondary" gutterBottom sx={{ mb: 3 }}>
        System overview and statistics
      </Typography>

      <Grid container spacing={3}>
        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <StatCard
            title="Total Users"
            value={stats.total_users}
            icon={<Users size={24} color="white" />}
            color="#3b82f6"
          />
        </Grid>

        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <StatCard
            title="Active Users"
            value={stats.active_users}
            icon={<UserCheck size={24} color="white" />}
            color="#22c55e"
          />
        </Grid>

        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <StatCard
            title="Admin Users"
            value={stats.admin_users}
            icon={<Shield size={24} color="white" />}
            color="#8b5cf6"
          />
        </Grid>

        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <StatCard
            title="Total Conversations"
            value={stats.total_conversations}
            icon={<MessageSquare size={24} color="white" />}
            color="#f59e0b"
          />
        </Grid>

        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <StatCard
            title="Total Messages"
            value={stats.total_messages}
            icon={<MessageSquare size={24} color="white" />}
            color="#ec4899"
          />
        </Grid>

        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <StatCard
            title="Active Last 24h"
            value={stats.users_last_24h}
            icon={<Activity size={24} color="white" />}
            color="#06b6d4"
          />
        </Grid>
      </Grid>

      <Typography variant="caption" color="textSecondary" sx={{ mt: 3, display: 'block' }}>
        Last updated: {new Date(stats.timestamp).toLocaleString()}
      </Typography>
    </Box>
  );
};
