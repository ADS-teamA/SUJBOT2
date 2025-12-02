/**
 * React Admin Application for SUJBOT2
 *
 * Provides:
 * - User management (CRUD)
 * - System health monitoring
 * - System statistics dashboard
 */

import { Admin, Resource, CustomRoutes, Layout, Menu } from 'react-admin';
import { Route, BrowserRouter } from 'react-router-dom';
import {
  Users,
  Activity,
  LayoutDashboard,
} from 'lucide-react';

import { dataProvider } from './dataProvider';
import { authProvider } from './authProvider';
import { UserList, UserEdit, UserCreate, UserShow } from './resources/users';
import { Dashboard } from './pages/Dashboard';
import { HealthPage } from './pages/HealthPage';
import { AdminLoginPage } from './pages/AdminLoginPage';

// Custom menu with icons
const AdminMenu = () => (
  <Menu>
    <Menu.DashboardItem primaryText="Dashboard" leftIcon={<LayoutDashboard size={20} />} />
    <Menu.ResourceItem name="users" primaryText="Users" leftIcon={<Users size={20} />} />
    <Menu.Item to="/health" primaryText="Health" leftIcon={<Activity size={20} />} />
  </Menu>
);

// Custom layout with menu
const AdminLayout = ({ children, ...props }: { children: React.ReactNode }) => (
  <Layout {...props} menu={AdminMenu}>
    {children}
  </Layout>
);

export function AdminApp() {
  // Debug: Log current URL
  console.log('AdminApp rendering, pathname:', window.location.pathname, 'href:', window.location.href);

  return (
    <BrowserRouter basename="/admin">
      <Admin
        dataProvider={dataProvider}
        authProvider={authProvider}
        loginPage={AdminLoginPage}
        layout={AdminLayout}
        dashboard={Dashboard}
      >
      <Resource
        name="users"
        list={UserList}
        edit={UserEdit}
        create={UserCreate}
        show={UserShow}
        options={{ label: 'Users' }}
      />

      <CustomRoutes>
        <Route path="/health" element={<HealthPage />} />
      </CustomRoutes>
    </Admin>
    </BrowserRouter>
  );
}
