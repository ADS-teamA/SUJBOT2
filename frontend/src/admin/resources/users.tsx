/**
 * User Resource for React Admin
 *
 * CRUD views for user management:
 * - UserList: Paginated table with filters
 * - UserEdit: Edit user form
 * - UserCreate: Create user form
 * - UserShow: User details view
 */

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
      {record.is_active ? 'Active' : 'Inactive'}
    </span>
  );
};

// Admin badge component
const AdminBadge = (_props: FieldProps) => {
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
      Admin
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

export const UserEdit = () => (
  <Edit>
    <SimpleForm>
      <TextInput source="id" disabled />
      <TextInput source="email" validate={[required(), email()]} />
      <PasswordInput
        source="password"
        label="New Password"
        helperText="Leave empty to keep current password"
      />
      <TextInput source="full_name" label="Full Name" />
      <BooleanInput source="is_active" label="Active" />
      <BooleanInput source="is_admin" label="Admin" />
      <SelectInput
        source="agent_variant"
        label="Agent Variant"
        choices={[
          { id: 'premium', name: 'Premium (Claude Opus/Sonnet)' },
          { id: 'cheap', name: 'Cheap (Claude Haiku)' },
          { id: 'local', name: 'Local (Llama 3.1 70B)' },
        ]}
      />
    </SimpleForm>
  </Edit>
);

export const UserCreate = () => (
  <Create>
    <SimpleForm>
      <TextInput source="email" validate={[required(), email()]} />
      <PasswordInput
        source="password"
        validate={[required(), minLength(8)]}
        helperText="Minimum 8 characters, must include uppercase, lowercase, number, and special character"
      />
      <TextInput source="full_name" label="Full Name" />
      <BooleanInput source="is_active" label="Active" defaultValue={true} />
      <BooleanInput source="is_admin" label="Admin" defaultValue={false} />
    </SimpleForm>
  </Create>
);

export const UserShow = () => (
  <Show>
    <SimpleShowLayout>
      <TextField source="id" />
      <EmailField source="email" />
      <TextField source="full_name" label="Full Name" />
      <BooleanField source="is_active" label="Active" />
      <BooleanField source="is_admin" label="Admin" />
      <TextField source="agent_variant" label="Agent Variant" />
      <DateField source="created_at" label="Created" showTime />
      <DateField source="updated_at" label="Updated" showTime />
      <DateField source="last_login_at" label="Last Login" showTime />
    </SimpleShowLayout>
  </Show>
);
