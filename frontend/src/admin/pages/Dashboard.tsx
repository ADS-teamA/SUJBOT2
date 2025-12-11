/**
 * Admin Dashboard
 *
 * Shows system statistics:
 * - Total users
 * - Active users
 * - Admin users
 * - Total conversations
 * - Total messages
 * - Users active in last 24h
 *
 * Spending statistics (CZK):
 * - Total spent across all users
 * - Average spent per message
 * - Average spent per conversation
 */

import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Card, CardContent, Typography, Box, Grid, CircularProgress, Divider } from '@mui/material';
import { Users, UserCheck, Shield, MessageSquare, Activity, Wallet } from 'lucide-react';

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
  total_spent_czk: number;
  avg_spent_per_message_czk: number;
  avg_spent_per_conversation_czk: number;
  timestamp: string;
}

interface StatCardProps {
  title: string;
  value: number;
  icon: React.ReactNode;
  color: string;
  suffix?: string;
  decimals?: number;
}

const StatCard = ({ title, value, icon, color, suffix, decimals }: StatCardProps) => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Box display="flex" alignItems="center" justifyContent="space-between">
        <Box>
          <Typography color="textSecondary" gutterBottom variant="body2">
            {title}
          </Typography>
          <Typography variant="h4" component="div">
            {decimals !== undefined ? value.toFixed(decimals) : value.toLocaleString()}
            {suffix && <Typography component="span" variant="h6" color="textSecondary"> {suffix}</Typography>}
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
  const { t } = useTranslation();
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
        <Typography color="error">{t('common.error')}: {error}</Typography>
      </Box>
    );
  }

  if (!stats) {
    return null;
  }

  return (
    <Box p={3}>
      <Typography variant="h4" gutterBottom>
        {t('admin.dashboard.title')}
      </Typography>
      <Typography variant="body2" color="textSecondary" gutterBottom sx={{ mb: 3 }}>
        {t('admin.dashboard.subtitle')}
      </Typography>

      <Grid container spacing={3}>
        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <StatCard
            title={t('admin.dashboard.totalUsers')}
            value={stats.total_users}
            icon={<Users size={24} color="white" />}
            color="#3b82f6"
          />
        </Grid>

        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <StatCard
            title={t('admin.dashboard.activeUsers')}
            value={stats.active_users}
            icon={<UserCheck size={24} color="white" />}
            color="#22c55e"
          />
        </Grid>

        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <StatCard
            title={t('admin.dashboard.adminUsers')}
            value={stats.admin_users}
            icon={<Shield size={24} color="white" />}
            color="#8b5cf6"
          />
        </Grid>

        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <StatCard
            title={t('admin.dashboard.totalConversations')}
            value={stats.total_conversations}
            icon={<MessageSquare size={24} color="white" />}
            color="#f59e0b"
          />
        </Grid>

        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <StatCard
            title={t('admin.dashboard.totalMessages')}
            value={stats.total_messages}
            icon={<MessageSquare size={24} color="white" />}
            color="#ec4899"
          />
        </Grid>

        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <StatCard
            title={t('admin.dashboard.activeLast24h')}
            value={stats.users_last_24h}
            icon={<Activity size={24} color="white" />}
            color="#06b6d4"
          />
        </Grid>
      </Grid>

      {/* Spending Statistics Section */}
      <Divider sx={{ my: 4 }} />
      <Typography variant="h5" gutterBottom>
        {t('admin.dashboard.spendingSection')}
      </Typography>
      <Typography variant="body2" color="textSecondary" gutterBottom sx={{ mb: 3 }}>
        {t('admin.dashboard.spendingSubtitle')}
      </Typography>

      <Grid container spacing={3}>
        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <StatCard
            title={t('admin.dashboard.totalSpent')}
            value={stats.total_spent_czk}
            icon={<Wallet size={24} color="white" />}
            color="#16a34a"
            suffix="Kč"
            decimals={2}
          />
        </Grid>

        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <StatCard
            title={t('admin.dashboard.avgPerMessage')}
            value={stats.avg_spent_per_message_czk}
            icon={<MessageSquare size={24} color="white" />}
            color="#0891b2"
            suffix="Kč"
            decimals={4}
          />
        </Grid>

        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <StatCard
            title={t('admin.dashboard.avgPerConversation')}
            value={stats.avg_spent_per_conversation_czk}
            icon={<MessageSquare size={24} color="white" />}
            color="#7c3aed"
            suffix="Kč"
            decimals={4}
          />
        </Grid>
      </Grid>

      <Typography variant="caption" color="textSecondary" sx={{ mt: 3, display: 'block' }}>
        {t('admin.dashboard.lastUpdated')}: {new Date(stats.timestamp).toLocaleString()}
      </Typography>
    </Box>
  );
};
