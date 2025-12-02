/**
 * User Resource for React Admin
 *
 * CRUD views for user management:
 * - UserList: Paginated table with filters
 * - UserEdit: Edit user form
 * - UserCreate: Create user form
 * - UserShow: User details view
 */

import { useTranslation } from 'react-i18next';
import {
  List,
  Datagrid,
  TextField,
  EmailField,
  BooleanField,
  DateField,
  Edit,
  Create,
  SimpleForm,
  TextInput,
  BooleanInput,
  SelectInput,
  PasswordInput,
  Show,
  SimpleShowLayout,
  EditButton,
  ShowButton,
  useRecordContext,
  required,
  email,
  minLength,
} from 'react-admin';

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

export const UserList = () => (
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
      <DateField source="created_at" label="Created" showTime />
      <DateField source="last_login_at" label="Last Login" showTime />
      <EditButton />
      <ShowButton />
    </Datagrid>
  </List>
);

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
        <DateField source="created_at" label={t('admin.users.createdAt')} showTime />
        <DateField source="updated_at" label={t('admin.users.updatedAt')} showTime />
        <DateField source="last_login_at" label={t('admin.users.lastLogin')} showTime />
      </SimpleShowLayout>
    </Show>
  );
};
