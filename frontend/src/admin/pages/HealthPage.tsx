/**
 * Health Monitoring Page
 *
 * Shows service health status:
 * - PostgreSQL database
 * - Backend API
 * - Other services as added
 */

import { useEffect, useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  CircularProgress,
  Button,
  Chip,
} from '@mui/material';
import { RefreshCw, CheckCircle, AlertTriangle, XCircle } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL !== undefined
  ? import.meta.env.VITE_API_BASE_URL
  : 'http://localhost:8000';

interface ServiceHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  latency_ms: number | null;
  message: string | null;
}

interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: ServiceHealth[];
  timestamp: string;
}

const StatusIcon = ({ status }: { status: string }) => {
  switch (status) {
    case 'healthy':
      return <CheckCircle size={24} color="#22c55e" />;
    case 'degraded':
      return <AlertTriangle size={24} color="#f59e0b" />;
    case 'unhealthy':
      return <XCircle size={24} color="#ef4444" />;
    default:
      return <AlertTriangle size={24} color="#6b7280" />;
  }
};

const StatusChip = ({ status }: { status: string }) => {
  const colors = {
    healthy: { bg: '#dcfce7', color: '#166534' },
    degraded: { bg: '#fef3c7', color: '#92400e' },
    unhealthy: { bg: '#fee2e2', color: '#991b1b' },
  };

  const style = colors[status as keyof typeof colors] || { bg: '#f3f4f6', color: '#374151' };

  return (
    <Chip
      label={status.charAt(0).toUpperCase() + status.slice(1)}
      sx={{
        backgroundColor: style.bg,
        color: style.color,
        fontWeight: 500,
      }}
    />
  );
};

const ServiceCard = ({ service }: { service: ServiceHealth }) => (
  <Card>
    <CardContent>
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
        <Box display="flex" alignItems="center" gap={2}>
          <StatusIcon status={service.status} />
          <Typography variant="h6">{service.name}</Typography>
        </Box>
        <StatusChip status={service.status} />
      </Box>

      {service.latency_ms !== null && (
        <Typography variant="body2" color="textSecondary">
          Latency: {service.latency_ms.toFixed(2)} ms
        </Typography>
      )}

      {service.message && (
        <Typography variant="body2" color="error" sx={{ mt: 1 }}>
          {service.message}
        </Typography>
      )}
    </CardContent>
  </Card>
);

export const HealthPage = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchHealth = async () => {
    try {
      setRefreshing(true);
      const response = await fetch(`${API_BASE_URL}/admin/health`, {
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error('Failed to fetch health status');
      }

      const data = await response.json();
      setHealth(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box p={3}>
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
        <Box>
          <Typography variant="h4" gutterBottom>
            System Health
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Monitor service status and performance
          </Typography>
        </Box>
        <Button
          variant="outlined"
          startIcon={<RefreshCw size={18} className={refreshing ? 'animate-spin' : ''} />}
          onClick={fetchHealth}
          disabled={refreshing}
        >
          Refresh
        </Button>
      </Box>

      {error && (
        <Box mb={3}>
          <Typography color="error">Error: {error}</Typography>
        </Box>
      )}

      {health && (
        <>
          <Box mb={3}>
            <Typography variant="h6" gutterBottom>
              Overall Status
            </Typography>
            <StatusChip status={health.status} />
          </Box>

          <Grid container spacing={3}>
            {health.services.map((service) => (
              <Grid size={{ xs: 12, sm: 6, md: 4 }} key={service.name}>
                <ServiceCard service={service} />
              </Grid>
            ))}
          </Grid>

          <Typography variant="caption" color="textSecondary" sx={{ mt: 3, display: 'block' }}>
            Last checked: {new Date(health.timestamp).toLocaleString()}
          </Typography>
        </>
      )}
    </Box>
  );
};
