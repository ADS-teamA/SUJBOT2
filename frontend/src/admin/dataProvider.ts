/**
 * React Admin Data Provider for SUJBOT2 Admin API
 *
 * Maps React Admin data operations to our backend API:
 * - GET /admin/users → getList, getMany
 * - GET /admin/users/{id} → getOne
 * - POST /admin/users → create
 * - PUT /admin/users/{id} → update
 * - DELETE /admin/users/{id} → delete
 */

import type { DataProvider } from 'react-admin';
import { fetchUtils } from 'react-admin';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL !== undefined
  ? import.meta.env.VITE_API_BASE_URL
  : 'http://localhost:8000';

const httpClient = (url: string, options: fetchUtils.Options = {}) => {
  return fetchUtils.fetchJson(url, {
    ...options,
    credentials: 'include', // Send httpOnly cookies
  });
};

export const dataProvider: DataProvider = {
  getList: async (resource, params) => {
    const page = params.pagination?.page ?? 1;
    const perPage = params.pagination?.perPage ?? 25;
    const offset = (page - 1) * perPage;

    const url = `${API_BASE_URL}/admin/${resource}?limit=${perPage}&offset=${offset}`;

    const { json } = await httpClient(url);

    return {
      data: json.users.map((user: Record<string, unknown>) => ({ ...user, id: user.id })),
      total: json.total,
    };
  },

  getOne: async (resource, params) => {
    const url = `${API_BASE_URL}/admin/${resource}/${params.id}`;
    const { json } = await httpClient(url);

    return { data: { ...json, id: json.id } };
  },

  getMany: async (resource, params) => {
    const results = await Promise.all(
      params.ids.map(id =>
        httpClient(`${API_BASE_URL}/admin/${resource}/${id}`)
      )
    );

    return {
      data: results.map(({ json }) => ({ ...json, id: json.id })),
    };
  },

  getManyReference: async (resource, params) => {
    const page = params.pagination?.page ?? 1;
    const perPage = params.pagination?.perPage ?? 25;
    const offset = (page - 1) * perPage;

    const url = `${API_BASE_URL}/admin/${resource}?limit=${perPage}&offset=${offset}`;
    const { json } = await httpClient(url);

    return {
      data: json.users.map((user: Record<string, unknown>) => ({ ...user, id: user.id })),
      total: json.total,
    };
  },

  create: async (resource, params) => {
    const url = `${API_BASE_URL}/admin/${resource}`;
    const { json } = await httpClient(url, {
      method: 'POST',
      body: JSON.stringify(params.data),
    });
    return { data: { ...json, id: json.id } };
  },

  update: async (resource, params) => {
    const url = `${API_BASE_URL}/admin/${resource}/${params.id}`;
    const { json } = await httpClient(url, {
      method: 'PUT',
      body: JSON.stringify(params.data),
    });

    return { data: { ...json, id: json.id } };
  },

  updateMany: async (resource, params) => {
    const results = await Promise.allSettled(
      params.ids.map(id =>
        httpClient(`${API_BASE_URL}/admin/${resource}/${id}`, {
          method: 'PUT',
          body: JSON.stringify(params.data),
        }).then(({ json }) => json.id)
      )
    );

    const successIds = results
      .filter((r): r is PromiseFulfilledResult<unknown> => r.status === 'fulfilled')
      .map(r => r.value);

    const failures = results.filter(r => r.status === 'rejected');
    if (failures.length > 0) {
      console.error('Some updates failed:', failures);
    }

    return { data: successIds };
  },

  delete: async (resource, params) => {
    const url = `${API_BASE_URL}/admin/${resource}/${params.id}`;
    await httpClient(url, { method: 'DELETE' });

    // React Admin expects the full record back
    if (!params.previousData) {
      console.error('Delete called without previousData - returning minimal object');
      return { data: { id: params.id } };
    }
    return { data: params.previousData };
  },

  deleteMany: async (resource, params) => {
    const results = await Promise.allSettled(
      params.ids.map(id =>
        httpClient(`${API_BASE_URL}/admin/${resource}/${id}`, {
          method: 'DELETE',
        }).then(() => id)
      )
    );

    const successIds = results
      .filter((r): r is PromiseFulfilledResult<unknown> => r.status === 'fulfilled')
      .map(r => r.value);

    const failures = results.filter(r => r.status === 'rejected');
    if (failures.length > 0) {
      console.error('Some deletes failed:', failures);
    }

    return { data: successIds };
  },
};
