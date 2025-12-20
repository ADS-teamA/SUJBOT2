/**
 * User Resource for React Admin
 *
 * CRUD views for user management:
 * - UserList: Paginated table with filters
 * - UserEdit: Edit user form
 * - UserCreate: Create user form
 * - UserShow: User details view
 */

import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  List,
  Datagrid,
  TextField,
  EmailField,
  BooleanField,
  DateField,
  NumberField,
  Edit,
  Create,
  SimpleForm,
  TextInput,
  BooleanInput,
  SelectInput,
  PasswordInput,
  NumberInput,
  Show,
  SimpleShowLayout,
  EditButton,
  ShowButton,
  useRecordContext,
  useRefresh,
  useNotify,
  required,
  email,
  minLength,
  Button,
  Confirm,
} from 'react-admin';
import { getApiBaseUrl } from '../dataProvider';
import { AdminConversationViewer } from '../components/AdminConversationViewer';

// Props for custom field components (React Admin passes label)
interface FieldProps {
  label?: string;
}

// Status chip component
const StatusField = (_props: FieldProps) => {
  const { t } = useTranslation();
  const record = useRecordContext();
  if (!record) return null;

  return (
    <span
      style={{
        padding: '4px 8px',
        borderRadius: '4px',
        fontSize: '12px',
        fontWeight: 500,
        backgroundColor: record.is_active ? '#dcfce7' : '#fee2e2',
        color: record.is_active ? '#166534' : '#991b1b',
      }}
    >
      {record.is_active ? t('admin.users.statusActive') : t('admin.users.statusInactive')}
    </span>
  );
};

// Admin badge component
const AdminBadge = (_props: FieldProps) => {
  const { t } = useTranslation();
  const record = useRecordContext();
  if (!record || !record.is_admin) return null;

  return (
    <span
      style={{
        padding: '4px 8px',
        borderRadius: '4px',
        fontSize: '12px',
        fontWeight: 500,
        backgroundColor: '#dbeafe',
        color: '#1e40af',
      }}
    >
      {t('admin.users.roleAdmin')}
    </span>
  );
};

// Spending field with color-coded progress bar
const SpendingField = (_props: FieldProps) => {
  const { t } = useTranslation();
  const record = useRecordContext();
  if (!record) return null;

  const spent = record.total_spent_czk ?? 0;
  const limit = record.spending_limit_czk ?? 500;
  const percentage = limit > 0 ? Math.min((spent / limit) * 100, 100) : 0;
  const isBlocked = spent >= limit;

  // Color based on percentage
  let barColor = '#22c55e'; // green
  let textColor = '#166534';
  if (percentage >= 90) {
    barColor = '#ef4444'; // red
    textColor = '#991b1b';
  } else if (percentage >= 70) {
    barColor = '#f59e0b'; // amber
    textColor = '#92400e';
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: '150px' }}>
      <div style={{ flex: 1 }}>
        <div
          style={{
            height: '8px',
            backgroundColor: '#e5e7eb',
            borderRadius: '4px',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              width: `${percentage}%`,
              height: '100%',
              backgroundColor: barColor,
              transition: 'width 0.3s ease',
            }}
          />
        </div>
        <div style={{ fontSize: '11px', color: textColor, marginTop: '2px' }}>
          {spent.toFixed(2)} / {limit.toFixed(2)} Kč
        </div>
      </div>
      {isBlocked && (
        <span
          style={{
            padding: '2px 6px',
            borderRadius: '4px',
            fontSize: '10px',
            fontWeight: 600,
            backgroundColor: '#fee2e2',
            color: '#991b1b',
          }}
        >
          {t('admin.users.blocked')}
        </span>
      )}
    </div>
  );
};

// Reset spending button for edit view
const ResetSpendingButton = () => {
  const { t } = useTranslation();
  const record = useRecordContext();
  const refresh = useRefresh();
  const notify = useNotify();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  if (!record) return null;

  const handleClick = () => setOpen(true);
  const handleDialogClose = () => setOpen(false);

  const handleConfirm = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('adminToken');
      const response = await fetch(
        `${getApiBaseUrl()}/admin/users/${record.id}/spending/reset`,
        {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        throw new Error('Failed to reset spending');
      }

      notify(t('admin.users.spendingResetSuccess'), { type: 'success' });
      refresh();
    } catch {
      notify(t('admin.users.spendingResetError'), { type: 'error' });
    } finally {
      setLoading(false);
      setOpen(false);
    }
  };

  return (
    <>
      <Button
        onClick={handleClick}
        disabled={loading || (record.total_spent_czk ?? 0) === 0}
        label={t('admin.users.resetSpending')}
        style={{
          backgroundColor: '#fef3c7',
          color: '#92400e',
          padding: '4px 12px',
          borderRadius: '4px',
        }}
      />
      <Confirm
        isOpen={open}
        loading={loading}
        title={t('admin.users.resetSpendingTitle')}
        content={t('admin.users.resetSpendingConfirm', { email: record.email })}
        onConfirm={handleConfirm}
        onClose={handleDialogClose}
      />
    </>
  );
};

export const UserList = () => {
  const { t } = useTranslation();
  return (
    <List
      sort={{ field: 'created_at', order: 'DESC' }}
      perPage={25}
    >
      <Datagrid rowClick="show">
        <TextField source="id" />
        <EmailField source="email" />
        <TextField source="full_name" label="Name" />
        <StatusField label="Status" />
        <AdminBadge label="Role" />
        <TextField source="agent_variant" label="Variant" />
        <SpendingField label={t('admin.users.spending')} />
        <DateField source="created_at" label="Created" showTime />
        <DateField source="last_login_at" label="Last Login" showTime />
        <EditButton />
        <ShowButton />
      </Datagrid>
    </List>
  );
};

export const UserEdit = () => {
  const { t } = useTranslation();
  return (
    <Edit>
      <SimpleForm>
        <TextInput source="id" disabled />
        <TextInput source="email" validate={[required(), email()]} label={t('admin.users.email')} />
        <PasswordInput
          source="password"
          label={t('admin.users.newPassword')}
          helperText={t('admin.users.passwordHelperEdit')}
        />
        <TextInput source="full_name" label={t('admin.users.fullName')} />
        <BooleanInput source="is_active" label={t('admin.users.isActive')} />
        <BooleanInput source="is_admin" label={t('admin.users.isAdmin')} />
        <SelectInput
          source="agent_variant"
          label={t('admin.users.agentVariant')}
          choices={[
            { id: 'premium', name: t('admin.users.variantPremium') },
            { id: 'cheap', name: t('admin.users.variantCheap') },
            { id: 'local', name: t('admin.users.variantLocal') },
          ]}
        />
        {/* Spending Management Section */}
        <div style={{ marginTop: '24px', marginBottom: '8px', fontWeight: 600, color: '#374151' }}>
          {t('admin.users.spendingSection')}
        </div>
        <NumberInput
          source="spending_limit_czk"
          label={t('admin.users.spendingLimit')}
          min={0}
          step={50}
          helperText={t('admin.users.spendingLimitHelper')}
        />
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginTop: '8px' }}>
          <SpendingField label={t('admin.users.currentSpending')} />
          <ResetSpendingButton />
        </div>
      </SimpleForm>
    </Edit>
  );
};

export const UserCreate = () => {
  const { t } = useTranslation();
  return (
    <Create>
      <SimpleForm>
        <TextInput source="email" validate={[required(), email()]} label={t('admin.users.email')} />
        <PasswordInput
          source="password"
          validate={[required(), minLength(8)]}
          label={t('admin.users.password')}
          helperText={t('admin.users.passwordHelperCreate')}
        />
        <TextInput source="full_name" label={t('admin.users.fullName')} />
        <BooleanInput source="is_active" label={t('admin.users.isActive')} defaultValue={true} />
        <BooleanInput source="is_admin" label={t('admin.users.isAdmin')} defaultValue={false} />
      </SimpleForm>
    </Create>
  );
};

/**
 * ConversationSection - Wrapper component to access record context for conversation viewer
 */
const ConversationSection = () => {
  const { t } = useTranslation();
  const record = useRecordContext();

  if (!record) return null;

  return (
    <div style={{ marginTop: '32px', gridColumn: '1 / -1' }}>
      <h3
        style={{
          fontSize: '16px',
          fontWeight: 600,
          marginBottom: '16px',
          color: '#374151',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}
      >
        {t('admin.conversations.sectionTitle')}
      </h3>
      <AdminConversationViewer userId={Number(record.id)} />
    </div>
  );
};

export const UserShow = () => {
  const { t } = useTranslation();
  return (
    <Show>
      <SimpleShowLayout>
        <TextField source="id" />
        <EmailField source="email" label={t('admin.users.email')} />
        <TextField source="full_name" label={t('admin.users.fullName')} />
        <BooleanField source="is_active" label={t('admin.users.isActive')} />
        <BooleanField source="is_admin" label={t('admin.users.isAdmin')} />
        <TextField source="agent_variant" label={t('admin.users.agentVariant')} />
        {/* Spending Information */}
        <NumberField source="spending_limit_czk" label={t('admin.users.spendingLimit')} />
        <SpendingField label={t('admin.users.spending')} />
        <DateField source="spending_reset_at" label={t('admin.users.spendingResetAt')} showTime />
        {/* Timestamps */}
        <DateField source="created_at" label={t('admin.users.createdAt')} showTime />
        <DateField source="updated_at" label={t('admin.users.updatedAt')} showTime />
        <DateField source="last_login_at" label={t('admin.users.lastLogin')} showTime />
        {/* User Conversation History */}
        <ConversationSection />
      </SimpleShowLayout>
    </Show>
  );
};
